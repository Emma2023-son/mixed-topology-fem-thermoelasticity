"""
MATCHED-DOF CONVERGENCE AND COST COMPARISON
======================================================
Based directly on the authoritative Validation core: validation.py.

Purpose
-------
1. Compare T3, Q4 and Hybrid on the SAME nodal meshes at each h.
2. Measure displacement L2 and H1-seminorm errors against a fine FE reference.
3. Measure the elastic energy seminorm error
       ||e||_E^2 = int_Omega (eps(e))^T D eps(e) dOmega.
4. Record mesh size, element counts, DOFs and wall-clock solution time.
5. Report observed local convergence orders only; no "optimal convergence" claim.
6. Optionally repeat the error calculation against a fine Hybrid reference as a
   reference-sensitivity check.

Important
---------
The H1 and L2 calculations use the corrected Phase-A implementation, which
compares FE fields at physical quadrature points rather than nearest nodes.
The energy seminorm is evaluated independently here using the same FE fields.

Run this script in the same directory as phaseA_corrected(4).py, or change
CORE_FILE below.
"""

from pathlib import Path
import importlib.util
import time
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
CORE_CANDIDATES = [
    BASE_DIR / "validation.py",
    BASE_DIR / "validation.py",
]
# CORE_CANDIDATES = [
#     "validation.py",
#     "validation.py",
# ]

H_LIST = [0.125, 0.0625, 0.03125, 0.015625]
ELEMENT_TYPES = ["T3", "Q4", "Hybrid"]

# Fine reference.  This is deliberately finer than the Phase-A campaign.
H_REF = 0.0078125
REFERENCE_ELEMENT_TYPE = "Q4"

# Optional reference sensitivity check.
RUN_HYBRID_REFERENCE_SENSITIVITY = True
HYBRID_REFERENCE_ELEMENT_TYPE = "Hybrid"

RESULTS_DIR = BASE_DIR / "Dof_cost_comparison_results"
FIGURES_DIR = BASE_DIR / "Dof_cost_comparison_figures"

# For timing, one call to the authoritative solver is timed per case.
# The timing includes mesh generation, assembly, the two linear solves and
# stress post-processing performed by solve_thermoelastic.


# =============================================================================
# LOAD PHASE-A CORE
# =============================================================================
def load_core():
    for candidate in CORE_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            spec = importlib.util.spec_from_file_location("validation_core_dof", path)
            fem = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fem)
            print(f"Loaded Validation core: {path.resolve()}")
            return fem
    raise FileNotFoundError(
        "Could not find validation.py. "
        "Place this script beside the Main Core core or edit CORE_CANDIDATES."
    )


fem = load_core()


# =============================================================================
# MESH HELPERS
# =============================================================================
def make_mesh(h, element_type):
    """Create the Phase-A mesh for the requested element topology."""
    nodes = fem.generate_nodes(h)

    if element_type == "T3":
        elements = fem.generate_triangles(nodes, h)
    elif element_type == "Q4":
        elements = fem.generate_quads(nodes, h)
    elif element_type == "Hybrid":
        t3, q4 = fem.generate_hybrid(nodes, h)
        elements = {"T3": t3, "Q4": q4}
    else:
        raise ValueError(f"Unknown element_type={element_type}")

    return np.asarray(nodes), elements


def element_counts(elements, element_type):
    if element_type == "Hybrid":
        return len(elements["T3"]), len(elements["Q4"])
    if element_type == "T3":
        return len(elements), 0
    return 0, len(elements)


# =============================================================================
# TIMED SOLVE
# =============================================================================
def solve_timed(h, element_type, nodes=None, elements=None):
    """Solve using the authoritative Phase-A solver and return wall time."""
    t0 = time.perf_counter()
    result = fem.solve_thermoelastic(
        h,
        element_type=element_type,
        nodes=nodes,
        elements=elements,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


# =============================================================================
# FAST REFERENCE-FIELD LOCATOR FOR ENERGY ERROR
# =============================================================================
class FEFieldLocator:
    """Locate a containing FE element and evaluate displacement/gradient."""

    def __init__(self, nodes, elements, U, element_type):
        self.nodes = np.asarray(nodes)
        self.elements = elements
        self.U = np.asarray(U)
        self.element_type = element_type
        self.flat = fem._flatten_elements(elements, element_type)
        self.centers = np.array([self.nodes[e].mean(axis=0) for e in self.flat])
        self.tree = cKDTree(self.centers)

    @staticmethod
    def _q4_local_coordinates(coords, point):
        xi = 0.0
        eta = 0.0
        point = np.asarray(point, dtype=float)

        for _ in range(15):
            N, dxi, deta = fem.q4_shape(xi, eta)
            mapped = np.array([
                N @ coords[:, 0],
                N @ coords[:, 1],
            ])
            residual = mapped - point

            if np.linalg.norm(residual) < 1e-11:
                break

            J = np.array([
                [dxi @ coords[:, 0], deta @ coords[:, 0]],
                [dxi @ coords[:, 1], deta @ coords[:, 1]],
            ])
            try:
                delta = np.linalg.solve(J, residual)
            except np.linalg.LinAlgError:
                return None

            xi -= delta[0]
            eta -= delta[1]

        if abs(xi) > 1.0 + 1e-8 or abs(eta) > 1.0 + 1e-8:
            return None
        return xi, eta

    def _candidate_indices(self, point):
        k = min(16, len(self.flat))
        _, cand = self.tree.query(np.asarray(point), k=k)
        return np.atleast_1d(cand)

    def _evaluate_element(self, e, point):
        coords = self.nodes[e]
        point = np.asarray(point, dtype=float)

        if len(e) == 3:
            N = fem._t3_shape_at_point(coords, point)
            if N is None:
                return None
            _, B, _ = fem.elasticity_tri_stiffness(
                coords, fem.E_MAT, fem.NU_MAT, fem.THICKNESS
            )
            dofs = np.array([d for n in e for d in (2*n, 2*n+1)], dtype=int)
            ue = self.U[dofs]
            return N @ ue.reshape(-1, 2), B @ ue

        N = fem._q4_shape_at_point(coords, point)
        if N is None:
            return None

        local = self._q4_local_coordinates(coords, point)
        if local is None:
            return None

        xi, eta = local
        _, B, _, _ = fem.q4_kinematics(coords, xi, eta)
        dofs = np.array([d for n in e for d in (2*n, 2*n+1)], dtype=int)
        ue = self.U[dofs]
        return N @ ue.reshape(-1, 2), B @ ue

    def evaluate(self, point):
        """Return (u, grad_u) at a physical point."""
        for j in self._candidate_indices(point):
            out = self._evaluate_element(self.flat[int(j)], point)
            if out is not None:
                return out

        # Robust fallback, needed occasionally for points lying on element
        # boundaries where nearest-center candidates can all fail a tolerance.
        for e in self.flat:
            out = self._evaluate_element(e, point)
            if out is not None:
                return out

        raise RuntimeError(f"Could not locate FE field at point {point}")


# =============================================================================
# QUADRATURE-BASED ELASTIC ENERGY ERROR
# =============================================================================
def engineering_strain(strain):
    """
    Return the engineering strain vector used by the Phase-A FEM core.

    In validation.py, the field evaluator returns

        B @ U = [eps_x, eps_y, gamma_xy]

    which is already a 3-component engineering strain vector.

    Therefore, do NOT index this quantity as strain[0,0], etc.
    """
    strain = np.asarray(strain, dtype=float).reshape(-1)

    if strain.size != 3:
        raise ValueError(
            "Validation strain evaluator must return three components "
            "[eps_x, eps_y, gamma_xy]. "
            f"Received shape={np.asarray(strain).shape}, "
            f"size={strain.size}."
        )

    return strain.copy()


def energy_error_against_reference(
    target_nodes,
    target_elements,
    target_U,
    target_element_type,
    ref_nodes,
    ref_elements,
    ref_U,
    ref_element_type,
):
    """
    Compute absolute and relative elastic energy-seminorm errors.

    ||e||_E^2 = integral (eps(e))^T D eps(e) dOmega.

    This is the elastic energy seminorm associated with the mechanical
    bilinear form. It is NOT the total thermoelastic potential energy.
    """
    D = fem.elasticity_D(fem.E_MAT, fem.NU_MAT)
    target_locator = FEFieldLocator(
        target_nodes, target_elements, target_U, target_element_type
    )
    ref_locator = FEFieldLocator(
        ref_nodes, ref_elements, ref_U, ref_element_type
    )

    err_sq = 0.0
    ref_sq = 0.0

    # Integrate over the reference mesh.  The target solution is evaluated
    # at exactly the same physical quadrature points.
    ref_flat = fem._flatten_elements(ref_elements, ref_element_type)

    for e in ref_flat:
        coords = ref_nodes[e]

        for p, w, N, B in fem._target_qps(coords):
            _, grad_ref = ref_locator.evaluate(p)
            _, grad_target = target_locator.evaluate(p)

            eps_ref = engineering_strain(grad_ref)
            eps_target = engineering_strain(grad_target)
            deps = eps_target - eps_ref

            err_sq += float(w) * float(deps @ D @ deps)
            ref_sq += float(w) * float(eps_ref @ D @ eps_ref)

    err = np.sqrt(max(err_sq, 0.0))
    ref_norm = np.sqrt(max(ref_sq, 0.0))
    rel = err / ref_norm if ref_norm > 0.0 else np.nan
    return err, rel


# The function above cannot use the tuple returned by solve_timed because the
# element type is metadata rather than a tuple entry. Use this explicit wrapper.
def compute_L2_H1(
    target_nodes,
    target_elements,
    target_U,
    target_element_type,
    ref_nodes,
    ref_elements,
    ref_U,
    ref_element_type,
):
    """
    Compute displacement L2 error and the Phase-A engineering-strain
    seminorm directly in DOF_COST.

    The target and reference meshes may have different element types.
    This avoids relying on Validation's single-element-type error routine.
    """

    # ------------------------------------------------------------
    # Build independent FE field evaluators.
    # ------------------------------------------------------------
    target_locator = FEFieldLocator(
        target_nodes,
        target_elements,
        target_U,
        target_element_type,
    )

    reference_locator = FEFieldLocator(
        ref_nodes,
        ref_elements,
        ref_U,
        ref_element_type,
    )

    err_L2_sq = 0.0
    err_H1_sq = 0.0

    # ------------------------------------------------------------
    # Integrate over the TARGET mesh.
    #
    # This is important: the error is evaluated at physical
    # quadrature points of the target FE space.
    # ------------------------------------------------------------
    target_flat = fem._flatten_elements(
        target_elements,
        target_element_type,
    )

    for e in target_flat:

        e = np.asarray(e, dtype=int)

        coords = target_nodes[e]

        # Target-element quadrature.
        qps = fem._target_qps(coords)

        for p, w, N, B in qps:

            # ----------------------------------------------------
            # Target FE field.
            # ----------------------------------------------------
            u_target, strain_target = target_locator.evaluate(p)

            # ----------------------------------------------------
            # Reference FE field evaluated at the SAME physical
            # point, using its OWN topology.
            # ----------------------------------------------------
            u_reference, strain_reference = reference_locator.evaluate(p)

            # ----------------------------------------------------
            # Displacement error.
            # ----------------------------------------------------
            du = (
                np.asarray(u_target, dtype=float).reshape(2)
                -
                np.asarray(u_reference, dtype=float).reshape(2)
            )

            err_L2_sq += float(w) * float(du @ du)

            # ----------------------------------------------------
            # Engineering-strain error.
            #
            # Phase-A's B matrix returns
            #
            # [eps_x, eps_y, gamma_xy].
            # ----------------------------------------------------
            eps_target = engineering_strain(strain_target)
            eps_reference = engineering_strain(strain_reference)

            deps = eps_target - eps_reference

            err_H1_sq += float(w) * float(deps @ deps)

    L2 = np.sqrt(max(err_L2_sq, 0.0))
    H1 = np.sqrt(max(err_H1_sq, 0.0))

    return L2, H1

# =============================================================================
# LOCAL OBSERVED ORDERS
# =============================================================================
def observed_orders(hs, errors):
    hs = np.asarray(hs, dtype=float)
    errors = np.asarray(errors, dtype=float)
    out = np.full(len(errors), np.nan)

    for i in range(1, len(errors)):
        if errors[i] > 0.0 and errors[i-1] > 0.0:
            out[i] = np.log(errors[i-1] / errors[i]) / np.log(hs[i-1] / hs[i])
    return out


# =============================================================================
# CASE EXECUTION
# =============================================================================
def run_primary_campaign():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("MATCHED-DOF CONVERGENCE AND COST COMPARISON")
    print("=" * 78)
    print(f"h levels       : {H_LIST}")
    print(f"reference h    : {H_REF}")
    print(f"reference type : {REFERENCE_ELEMENT_TYPE}")
    print("Matched-node meshes are generated once per h and reused by all methods.")

    # -------------------------------------------------------------------------
    # Build one common nodal mesh at each h.
    # -------------------------------------------------------------------------
    mesh_by_h = {}
    for h in H_LIST + [H_REF]:
        mesh_by_h[h] = fem.generate_nodes(h)

    # -------------------------------------------------------------------------
    # Primary reference.
    # -------------------------------------------------------------------------
    ref_nodes = mesh_by_h[H_REF]
    ref_elements = fem.generate_quads(ref_nodes, H_REF)
    reference, ref_time = solve_timed(
        H_REF,
        REFERENCE_ELEMENT_TYPE,
        nodes=ref_nodes,
        elements=ref_elements,
    )

    print(
        f"\nReference: {REFERENCE_ELEMENT_TYPE}, h={H_REF:g}, "
        f"nodes={len(ref_nodes)}, time={ref_time:.6f} s"
    )

    primary_rows = []
    solution_cache = {}

    # -------------------------------------------------------------------------
    # Main matched-DOF campaign.
    # -------------------------------------------------------------------------
    for h in H_LIST:
        common_nodes = mesh_by_h[h]
        n_nodes = len(common_nodes)

        print(f"\n--- h = {h:g}; common nodes = {n_nodes} ---")

        for etype in ELEMENT_TYPES:
            if etype == "T3":
                elements = fem.generate_triangles(common_nodes, h)
            elif etype == "Q4":
                elements = fem.generate_quads(common_nodes, h)
            else:
                t3, q4 = fem.generate_hybrid(common_nodes, h)
                elements = {"T3": t3, "Q4": q4}

            # Guard against accidental topology-dependent nodal sets.
            if len(common_nodes) != n_nodes:
                raise RuntimeError("Matched-node invariant violated.")

            result, elapsed = solve_timed(
                h,
                etype,
                nodes=common_nodes,
                elements=elements,
            )
            solution_cache[(etype, h)] = result

            # Result is (nodes,elements,T,U,disp,stresses,vmises,centroids).
            L2, H1 = compute_L2_H1(
                target_nodes=result[0],
                target_elements=result[1],
                target_U=result[3],
                target_element_type=etype,
                ref_nodes=reference[0],
                ref_elements=reference[1],
                ref_U=reference[3],
                ref_element_type=REFERENCE_ELEMENT_TYPE,
            )
            
            E_abs, E_rel = energy_error_against_reference(
                result[0], result[1], result[3], etype,
                reference[0], reference[1], reference[3], REFERENCE_ELEMENT_TYPE,
            )

            n_t3, n_q4 = element_counts(elements, etype)
            n_elements = n_t3 + n_q4
            mech_dofs = 2 * n_nodes
            thermal_dofs = n_nodes

            row = {
                "element_type": etype,
                "h": h,
                "n_nodes": n_nodes,
                "thermal_dofs": thermal_dofs,
                "mechanical_dofs": mech_dofs,
                "total_sequential_unknowns": thermal_dofs + mech_dofs,
                "n_elements": n_elements,
                "n_T3": n_t3,
                "n_Q4": n_q4,
                "wall_time_s": elapsed,
                "L2_error": L2,
                "H1_seminorm_error": H1,
                "energy_error": E_abs,
                "relative_energy_error": E_rel,
                "reference_type": REFERENCE_ELEMENT_TYPE,
                "reference_h": H_REF,
            }
            primary_rows.append(row)

            print(
                f"{etype:6s} | elements={n_elements:6d} "
                f"(T3={n_t3:5d}, Q4={n_q4:5d}) | "
                f"DOF_m={mech_dofs:6d} | time={elapsed:9.4f}s | "
                f"L2={L2:.4e} | H1={H1:.4e} | Erel={E_rel:.4e}"
            )

    df = pd.DataFrame(primary_rows)

    # -------------------------------------------------------------------------
    # Add observed local orders.
    # -------------------------------------------------------------------------
    for etype in ELEMENT_TYPES:
        mask = df["element_type"] == etype
        idx = df.index[mask]
        sub = df.loc[idx].sort_values("h")
        # Reorder coarsest -> finest for conventional rate reporting.
        sub = sub.sort_values("h", ascending=False)
        for col, outcol in [
            ("L2_error", "L2_order"),
            ("H1_seminorm_error", "H1_order"),
            ("energy_error", "energy_order"),
        ]:
            rates = observed_orders(sub["h"].values, sub[col].values)
            df.loc[sub.index, outcol] = rates

    df = df.sort_values(["element_type", "h"], ascending=[True, False])
    primary_csv = RESULTS_DIR / "primary_results.csv"
    df.to_csv(primary_csv, index=False, float_format="%.10e")

    # -------------------------------------------------------------------------
    # Optional Hybrid-reference sensitivity.
    # -------------------------------------------------------------------------
    hybrid_ref = None
    if RUN_HYBRID_REFERENCE_SENSITIVITY:
        print("\n" + "-" * 78)
        print("REFERENCE SENSITIVITY: fine Hybrid reference")
        print("-" * 78)

        ref_nodes_h = mesh_by_h[H_REF]
        t3r, q4r = fem.generate_hybrid(ref_nodes_h, H_REF)
        ref_elements_h = {"T3": t3r, "Q4": q4r}
        hybrid_ref, hybrid_ref_time = solve_timed(
            H_REF,
            HYBRID_REFERENCE_ELEMENT_TYPE,
            nodes=ref_nodes_h,
            elements=ref_elements_h,
        )

        sensitivity_rows = []
        for etype in ELEMENT_TYPES:
            for h in H_LIST:
                result = solution_cache[(etype, h)]
                L2, H1 = compute_L2_H1(
                    target_nodes=result[0],
                    target_elements=result[1],
                    target_U=result[3],
                    target_element_type=etype,
                    ref_nodes=hybrid_ref[0],
                    ref_elements=hybrid_ref[1],
                    ref_U=hybrid_ref[3],
                    ref_element_type=HYBRID_REFERENCE_ELEMENT_TYPE,
                )
                E_abs, E_rel = energy_error_against_reference(
                    result[0], result[1], result[3], etype,
                    hybrid_ref[0], hybrid_ref[1], hybrid_ref[3],
                    HYBRID_REFERENCE_ELEMENT_TYPE,
                )
                sensitivity_rows.append({
                    "element_type": etype,
                    "h": h,
                    "L2_error_hybrid_ref": L2,
                    "H1_seminorm_error_hybrid_ref": H1,
                    "energy_error_hybrid_ref": E_abs,
                    "relative_energy_error_hybrid_ref": E_rel,
                    "reference_type": HYBRID_REFERENCE_ELEMENT_TYPE,
                    "reference_h": H_REF,
                })

        sens = pd.DataFrame(sensitivity_rows)
        sens.to_csv(
            RESULTS_DIR / "hybrid_reference_sensitivity.csv",
            index=False,
            float_format="%.10e",
        )

        print(
            f"Hybrid reference solved: h={H_REF:g}, "
            f"nodes={len(ref_nodes_h)}, time={hybrid_ref_time:.6f} s"
        )

    # -------------------------------------------------------------------------
    # Matched-DOF audit.
    # -------------------------------------------------------------------------
    audit = (
        df.groupby("h")["n_nodes"]
        .agg(["min", "max", "nunique"])
        .reset_index()
    )
    audit.to_csv(RESULTS_DIR / "matched_dof_audit.csv", index=False)

    if np.any(audit["nunique"].values != 1):
        raise RuntimeError("Matched-DOF audit failed: node counts differ by method.")

    # -------------------------------------------------------------------------
    # Summary tables.
    # -------------------------------------------------------------------------
    summary_cols = [
        "element_type", "h", "n_nodes", "mechanical_dofs",
        "n_elements", "n_T3", "n_Q4", "wall_time_s",
        "L2_error", "L2_order", "H1_seminorm_error", "H1_order",
        "energy_error", "energy_order", "relative_energy_error",
    ]
    df[summary_cols].to_csv(
        RESULTS_DIR / "summary_table.csv",
        index=False,
        float_format="%.10e",
    )

    print("\n" + "=" * 78)
    print("PHASE C COMPLETED")
    print("=" * 78)
    print(f"Primary results : {primary_csv.resolve()}")
    print(f"Results folder  : {RESULTS_DIR.resolve()}")
    print(f"Figures folder  : {FIGURES_DIR.resolve()}")

    return df, solution_cache, reference, hybrid_ref


# =============================================================================
# PLOTS
# =============================================================================
def make_plots(df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("L2_error", "L2 displacement error", "L2 displacement error"),
        ("H1_seminorm_error", "H1 seminorm", "H1-seminorm error"),
        ("energy_error", "Elastic energy seminorm error", "Elastic energy error"),
    ]

    for col, ylabel, title in metrics:
        plt.figure(figsize=(7, 5))
        for etype in ELEMENT_TYPES:
            sub = df[df["element_type"] == etype].sort_values("h")
            plt.loglog(sub["h"], sub[col], "o-", label=etype)
        plt.xlabel("h")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{col}_vs_h.png", dpi=300)
        plt.close()

    # Error versus wall-clock time is a useful computational-efficiency plot.
    for col, ylabel, filename in [
        ("L2_error", "L2 displacement error", "L2_vs_time.png"),
        ("H1_seminorm_error", "H1-seminorm error", "H1_vs_time.png"),
        ("energy_error", "Elastic energy error", "energy_vs_time.png"),
    ]:
        plt.figure(figsize=(7, 5))
        for etype in ELEMENT_TYPES:
            sub = df[df["element_type"] == etype].sort_values("wall_time_s")
            plt.loglog(sub["wall_time_s"], sub[col], "o-", label=etype)
        plt.xlabel("Wall-clock time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} versus wall-clock time")
        plt.grid(True, which="both", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, dpi=300)
        plt.close()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df, solution_cache, reference, hybrid_ref = run_primary_campaign()
    make_plots(df)
