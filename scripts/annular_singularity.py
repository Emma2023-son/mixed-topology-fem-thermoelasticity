"""
ANNULAR, ELEMENT-INTEGRATED CORNER-SINGULARITY ANALYSIS
====================================================================

Purpose
-------
Replace ray-by-ray point sampling by an angularly averaged, element-integrated
measure of the stress field around the re-entrant corner c=(0.5,0.5).

For annulus A(r1,r2) inside the local 270-degree wedge, compute

    S_vm = [ int_A sigma_vm^2 dA / int_A dA ]^(1/2)

and the compliance-weighted stress measure

    S_E  = [ int_A sigma^T D^{-1} sigma dA / int_A dA ]^(1/2).

If sigma ~ r^{-beta}, both RMS measures scale as r^{-beta}. Therefore we fit

    log S = a - beta log r,

using predeclared annular shells only. No best-window/R^2 selection is used.

Important interpretation
------------------------
The classical L-shaped elasticity displacement exponent is lambda_u≈0.54448,
so the associated stress exponent is beta_sigma=1-lambda_u≈0.45552.
This script tests whether the computed thermoelastic stress field approaches
that local stress scaling; it does not assume that it must.

Numerical integration
---------------------
* T3: 7-point Dunavant quadrature (stress strain part is constant but thermal
  stress varies through the interpolated temperature).
* Q4: 4x4 Gauss-Legendre quadrature.
* Hybrid: each element is integrated with its native rule.
* Annular clipping is done at quadrature points. A shell-area coverage ratio
  against the exact 270-degree annular-sector area is exported as a diagnostic.

Run this script from the same directory as validation.py.
"""

from pathlib import Path
import importlib.util
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import linregress


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CORE_FILE  =  BASE_DIR / "validation.py"


CORNER = np.array([0.5, 0.5], dtype=float)
OMEGA = 3.0 * np.pi / 4.0  # half-angle notation used in characteristic eqn
WEDGE_ANGLE = 3.0 * np.pi / 2.0  # physical interior angle = 270 degrees

H_LIST = [0.125, 0.0625, 0.03125, 0.015625]
ELEMENT_TYPES = ["T3", "Q4", "Hybrid"]

# Shell edges are fixed a priori in units of h.  The nonuniform sequence gives
# more shells without making the annuli too narrow near the corner.
SHELL_EDGES_OVER_H = np.array([1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])

# Restrict to the purely local 270-degree wedge before the outer square boundary
# starts truncating the annulus.  The nearest outer-boundary distance is 0.5.
R_PHYSICAL_MAX = 0.45

# Principal fit is predeclared.  Shells must lie wholly in this range.
PRINCIPAL_RMIN_OVER_H = 2.0
PRINCIPAL_RMAX_OVER_H = 12.0

# Diagnostics / admissibility thresholds.  These do not choose a 'best' fit.
MIN_DISTINCT_ELEMENTS = {"T3": 8, "Q4": 6, "Hybrid": 8}
MIN_QUAD_POINTS = 18
MIN_AREA_COVERAGE = 0.60
MIN_SHELLS_FOR_FIT = 3

RESULTS_DIR = Path("annular_singularity_results")
FIGURES_DIR = Path("annular_singularity_figures")


# -----------------------------------------------------------------------------
# LOAD PHASE-A CORE
# -----------------------------------------------------------------------------
def load_core(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path!s}. Put this script beside validation.py."
        )
    spec = importlib.util.spec_from_file_location("validation_core_singularity", path)
    fem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fem)
    return fem


fem = load_core(CORE_FILE)


# -----------------------------------------------------------------------------
# THEORY
# -----------------------------------------------------------------------------
def theoretical_lambda(omega=OMEGA):
    f = lambda lam: lam * np.sin(2.0 * omega) + np.sin(2.0 * lam * omega)
    return brentq(f, 0.5, 0.6)


LAMBDA_THEORY = theoretical_lambda()
BETA_THEORY = 1.0 - LAMBDA_THEORY


# -----------------------------------------------------------------------------
# CONSTITUTIVE MATRIX
# -----------------------------------------------------------------------------
def plane_stress_D(E, nu):
    return E / (1.0 - nu**2) * np.array(
        [[1.0, nu, 0.0],
         [nu, 1.0, 0.0],
         [0.0, 0.0, 0.5 * (1.0 - nu)]],
        dtype=float,
    )


D = plane_stress_D(fem.E_MAT, fem.NU_MAT)
D_INV = np.linalg.inv(D)


# -----------------------------------------------------------------------------
# QUADRATURE
# -----------------------------------------------------------------------------
def t3_dunavant_7(coords):
    """Degree-5 seven-point Dunavant rule, physical weights summing to area."""
    coords = np.asarray(coords, dtype=float)
    area = 0.5 * abs(np.linalg.det(np.array([
        coords[1] - coords[0],
        coords[2] - coords[0],
    ])))

    # Barycentric coordinates and normalized weights (sum = 1).
    data = [
        ((1/3, 1/3, 1/3), 0.225000000000000),
    ]
    a1 = 0.059715871789770
    b1 = 0.470142064105115
    w1 = 0.132394152788506
    data += [
        ((a1, b1, b1), w1),
        ((b1, a1, b1), w1),
        ((b1, b1, a1), w1),
    ]
    a2 = 0.797426985353087
    b2 = 0.101286507323456
    w2 = 0.125939180544827
    data += [
        ((a2, b2, b2), w2),
        ((b2, a2, b2), w2),
        ((b2, b2, a2), w2),
    ]

    out = []
    for bary, wn in data:
        N = np.asarray(bary, dtype=float)
        p = N @ coords
        out.append((p, area * wn, N))
    return out


def q4_gauss_4(coords):
    """4x4 tensor-product Gauss-Legendre rule in physical coordinates."""
    coords = np.asarray(coords, dtype=float)
    gps, gws = np.polynomial.legendre.leggauss(4)
    out = []
    for xi, wx in zip(gps, gws):
        for eta, wy in zip(gps, gws):
            N, B, detJ, _ = fem.q4_kinematics(coords, xi, eta)
            if detJ <= 0:
                raise RuntimeError("Non-positive Q4 Jacobian encountered.")
            p = N @ coords
            out.append((p, wx * wy * detJ, N, B))
    return out


# -----------------------------------------------------------------------------
# ELEMENT STRESS SAMPLING
# -----------------------------------------------------------------------------
def element_records(nodes, elements, T, U, element_type):
    """Yield quadrature-level stress records with element IDs and physical weights."""
    records = []

    def process_t3(e, eid, global_offset=0):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        dofs = np.array([d for n in e for d in (2*n, 2*n + 1)], dtype=int)
        ue = U[dofs]
        _, B, _ = fem.elasticity_tri_stiffness(
            coords, fem.E_MAT, fem.NU_MAT, fem.THICKNESS
        )
        strain = B @ ue
        for p, w, N in t3_dunavant_7(coords):
            Tgp = float(N @ T[e])
            eps_th = fem.ALPHA_TH * (Tgp - fem.T0) * np.array([1.0, 1.0, 0.0])
            sigma = D @ (strain - eps_th)
            vm = float(fem.von_mises(sigma))
            energy_density = float(sigma @ D_INV @ sigma)
            records.append({
                "point": p, "weight": float(w), "sigma": sigma,
                "sigma_vm": vm, "sigma_energy": energy_density,
                "element_uid": f"T3:{global_offset + eid}", "kind": "T3",
            })

    def process_q4(e, eid, global_offset=0):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        dofs = np.array([d for n in e for d in (2*n, 2*n + 1)], dtype=int)
        ue = U[dofs]
        for p, w, N, B in q4_gauss_4(coords):
            strain = B @ ue
            Tgp = float(N @ T[e])
            eps_th = fem.ALPHA_TH * (Tgp - fem.T0) * np.array([1.0, 1.0, 0.0])
            sigma = D @ (strain - eps_th)
            vm = float(fem.von_mises(sigma))
            energy_density = float(sigma @ D_INV @ sigma)
            records.append({
                "point": p, "weight": float(w), "sigma": sigma,
                "sigma_vm": vm, "sigma_energy": energy_density,
                "element_uid": f"Q4:{global_offset + eid}", "kind": "Q4",
            })

    if element_type == "T3":
        for eid, e in enumerate(elements):
            process_t3(e, eid)
    elif element_type == "Q4":
        for eid, e in enumerate(elements):
            process_q4(e, eid)
    elif element_type == "Hybrid":
        t3 = elements["T3"]
        q4 = elements["Q4"]
        for eid, e in enumerate(t3):
            process_t3(e, eid)
        for eid, e in enumerate(q4):
            process_q4(e, eid)
    else:
        raise ValueError(element_type)

    return records


# -----------------------------------------------------------------------------
# ANNULAR AGGREGATION
# -----------------------------------------------------------------------------
def exact_sector_area(rin, rout):
    return 0.5 * WEDGE_ANGLE * (rout**2 - rin**2)


def aggregate_shells(records, h, element_type):
    rows = []
    for a, b in zip(SHELL_EDGES_OVER_H[:-1], SHELL_EDGES_OVER_H[1:]):
        rin = a * h
        rout = b * h

        # Only retain fully local shells, avoiding outer-square truncation.
        physically_local = rout <= R_PHYSICAL_MAX + 1e-14

        selected = []
        for rec in records:
            r = float(np.linalg.norm(rec["point"] - CORNER))
            if rin <= r < rout:
                selected.append((rec, r))

        if selected:
            area_q = sum(rec["weight"] for rec, _ in selected)
            vm2 = sum(rec["weight"] * rec["sigma_vm"]**2 for rec, _ in selected)
            e2 = sum(rec["weight"] * rec["sigma_energy"] for rec, _ in selected)

            sxx2 = sum(rec["weight"] * rec["sigma"][0]**2 for rec, _ in selected)
            syy2 = sum(rec["weight"] * rec["sigma"][1]**2 for rec, _ in selected)
            txy2 = sum(rec["weight"] * rec["sigma"][2]**2 for rec, _ in selected)

            S_vm = math.sqrt(vm2 / area_q)
            S_E = math.sqrt(e2 / area_q)
            S_xx = math.sqrt(sxx2 / area_q)
            S_yy = math.sqrt(syy2 / area_q)
            S_xy = math.sqrt(txy2 / area_q)
            nelem = len({rec["element_uid"] for rec, _ in selected})
            nq = len(selected)
        else:
            area_q = S_vm = S_E = S_xx = S_yy = S_xy = np.nan
            nelem = nq = 0

        area_exact = exact_sector_area(rin, rout)
        coverage = area_q / area_exact if np.isfinite(area_q) and area_exact > 0 else 0.0

        resolved = (
            physically_local
            and nelem >= MIN_DISTINCT_ELEMENTS[element_type]
            and nq >= MIN_QUAD_POINTS
            and coverage >= MIN_AREA_COVERAGE
            and np.isfinite(S_vm)
            and S_vm > 0
            and np.isfinite(S_E)
            and S_E > 0
        )

        rows.append({
            "element_type": element_type,
            "h": h,
            "rin_over_h": a,
            "rout_over_h": b,
            "rin": rin,
            "rout": rout,
            "r_rep": math.sqrt(rin * rout),
            "r_rep_over_h": math.sqrt(a * b),
            "physically_local": physically_local,
            "distinct_elements": nelem,
            "quad_points": nq,
            "quadrature_area": area_q,
            "exact_sector_area": area_exact,
            "area_coverage": coverage,
            "resolved_shell": resolved,
            "S_vm": S_vm,
            "S_energy": S_E,
            "S_sigma_xx": S_xx,
            "S_sigma_yy": S_yy,
            "S_tau_xy": S_xy,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# POWER-LAW FITS
# -----------------------------------------------------------------------------
def fit_measure(df, column):
    d = df[
        df["resolved_shell"]
        & (df["rin_over_h"] >= PRINCIPAL_RMIN_OVER_H - 1e-12)
        & (df["rout_over_h"] <= PRINCIPAL_RMAX_OVER_H + 1e-12)
        & np.isfinite(df[column])
        & (df[column] > 0)
    ].copy()

    if len(d) < MIN_SHELLS_FOR_FIT:
        return {
            "measure": column, "n_shells": len(d), "beta_hat": np.nan,
            "slope": np.nan, "intercept": np.nan, "R2": np.nan,
            "stderr_slope": np.nan, "rmin": np.nan, "rmax": np.nan,
        }

    x = np.log(d["r_rep"].to_numpy())
    y = np.log(d[column].to_numpy())
    reg = linregress(x, y)
    beta = -reg.slope
    return {
        "measure": column,
        "n_shells": len(d),
        "beta_hat": beta,
        "slope": reg.slope,
        "intercept": reg.intercept,
        "R2": reg.rvalue**2,
        "stderr_slope": reg.stderr,
        "rmin": d["rin"].min(),
        "rmax": d["rout"].max(),
    }


def fit_all_measures(shell_df):
    measures = ["S_vm", "S_energy", "S_sigma_xx", "S_sigma_yy", "S_tau_xy"]
    rows = []
    for (etype, h), group in shell_df.groupby(["element_type", "h"], sort=False):
        for measure in measures:
            row = fit_measure(group, measure)
            row.update({
                "element_type": etype,
                "h": h,
                "lambda_theory": LAMBDA_THEORY,
                "beta_theory": BETA_THEORY,
            })
            if np.isfinite(row["beta_hat"]):
                row["abs_beta_error"] = abs(row["beta_hat"] - BETA_THEORY)
                row["rel_beta_error_pct"] = 100.0 * row["abs_beta_error"] / BETA_THEORY
            else:
                row["abs_beta_error"] = np.nan
                row["rel_beta_error_pct"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# CAMPAIGN
# -----------------------------------------------------------------------------
def run_campaign():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    print("\nANNULAR SINGULARITY ANALYSIS")
    print("=" * 68)
    print(f"lambda_theory = {LAMBDA_THEORY:.12f}")
    print(f"beta_theory   = {BETA_THEORY:.12f}")
    print(f"principal window = [{PRINCIPAL_RMIN_OVER_H:g}h, {PRINCIPAL_RMAX_OVER_H:g}h]")
    print(f"physical locality cap r <= {R_PHYSICAL_MAX:g}\n")

    all_shells = []

    for etype in ELEMENT_TYPES:
        for h in H_LIST:
            print(f"Solving {etype:7s} h={h:.6f} ...", flush=True)
            out = fem.solve_thermoelastic(h, etype)
            nodes, elements, T, U, *extra = out

            print(
                f"  nodes={len(nodes):6d}, Tmin={np.min(T):.6f}, Tmax={np.max(T):.6f}",
                flush=True,
            )
            records = element_records(nodes, elements, T, U, etype)
            shells = aggregate_shells(records, h, etype)
            all_shells.append(shells)

            nresolved = int(shells["resolved_shell"].sum())
            print(f"  quadrature records={len(records):7d}, resolved local shells={nresolved}")

    shell_df = pd.concat(all_shells, ignore_index=True)
    fit_df = fit_all_measures(shell_df)

    shell_csv = RESULTS_DIR / "phaseB4_annular_shells.csv"
    fit_csv = RESULTS_DIR / "phaseB4_principal_fits.csv"
    shell_df.to_csv(shell_csv, index=False)
    fit_df.to_csv(fit_csv, index=False)

    principal = fit_df[fit_df["measure"].isin(["S_vm", "S_energy"])].copy()
    print("\nPrincipal annular fits")
    print("-" * 68)
    print(principal[[
        "element_type", "h", "measure", "n_shells", "beta_hat", "R2",
        "rel_beta_error_pct"
    ]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Wide manuscript-oriented table.
    wide = principal.pivot_table(
        index=["element_type", "h"], columns="measure",
        values=["beta_hat", "R2", "n_shells", "rel_beta_error_pct"],
        aggfunc="first"
    )
    wide.to_csv(RESULTS_DIR / "manuscript_summary.csv")

    make_figures(shell_df, fit_df)

    print("\nSaved:")
    print(f"  {shell_csv}")
    print(f"  {fit_csv}")
    print(f"  {RESULTS_DIR / 'manuscript_summary.csv'}")
    print(f"  figures -> {FIGURES_DIR}")

    return shell_df, fit_df


# -----------------------------------------------------------------------------
# FIGURES
# -----------------------------------------------------------------------------
def make_figures(shell_df, fit_df):
    # One figure per element type: annular von-Mises RMS curves by mesh level.
    for etype in ELEMENT_TYPES:
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        for h in H_LIST:
            d = shell_df[
                (shell_df["element_type"] == etype)
                & shell_df["resolved_shell"]
                & np.isfinite(shell_df["S_vm"])
            ]
            d = d[d["h"] == h]
            if len(d):
                ax.loglog(d["r_rep"], d["S_vm"], marker="o", label=f"h={h:g}")

        # Theory reference slope, normalized to the finest available midpoint.
        dref = shell_df[
            (shell_df["element_type"] == etype)
            & shell_df["resolved_shell"]
            & np.isfinite(shell_df["S_vm"])
        ].sort_values("r_rep")
        if len(dref):
            rr = np.array([dref["r_rep"].min(), dref["r_rep"].max()])
            anchor = dref.iloc[len(dref)//2]
            c = anchor["S_vm"] * anchor["r_rep"]**BETA_THEORY
            ax.loglog(rr, c * rr**(-BETA_THEORY), "--",
                      label=rf"theory slope $-\beta$, $\beta={BETA_THEORY:.4f}$")

        ax.set_xlabel("distance from re-entrant corner, r")
        ax.set_ylabel(r"annular RMS von Mises stress, $S_{vm}$")
        ax.set_title(f"Annular stress scaling — {etype}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"phaseB4_annular_vm_{etype}.png", dpi=300)
        plt.close(fig)

    # Mesh-refinement plot of beta estimates for the two principal measures.
    for measure in ["S_vm", "S_energy"]:
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        for etype in ELEMENT_TYPES:
            d = fit_df[
                (fit_df["element_type"] == etype)
                & (fit_df["measure"] == measure)
                & np.isfinite(fit_df["beta_hat"])
            ].sort_values("h", ascending=False)
            if len(d):
                ax.semilogx(d["h"], d["beta_hat"], marker="o", label=etype)
        ax.axhline(BETA_THEORY, linestyle="--", label=f"theory {BETA_THEORY:.4f}")
        ax.invert_xaxis()
        ax.set_xlabel("mesh size h (finer to the right)")
        ax.set_ylabel(r"estimated stress exponent $\hat\beta$")
        ax.set_title(f"Exponent refinement — {measure}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"phaseB4_beta_refinement_{measure}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    run_campaign()
