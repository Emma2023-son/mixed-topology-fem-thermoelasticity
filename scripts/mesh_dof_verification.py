"""
Complete mesh/DOF reporting and independent manufactured thermoelastic verification.

Uses the authoritative Validation FEM core without modifying it.

Outputs
-------
phaseD2_results_corrected/
    lshape_mesh_dof_table.csv
    lshape_mesh_dof_table_manuscript.csv
    manufactured_verification_results.csv
    manufactured_verification_summary.csv
    manufactured_verification_inputs.csv
"""

from pathlib import Path
import importlib.util
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# -----------------------------------------------------------------------------
# Load authoritative Phase-A core
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CORE_CANDIDATES = [
    BASE_DIR / "validation.py",
    BASE_DIR / "validation.py",
]
CORE = next((p for p in CORE_CANDIDATES if p.exists()), None)
if CORE is None:
    raise FileNotFoundError("Could not find validation.py")

spec = importlib.util.spec_from_file_location("fem_core_verification", CORE)
fem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fem)

OUT_DIR = BASE_DIR / "Mesh_DOF_comparison_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Production L-shaped meshes
# -----------------------------------------------------------------------------
H_LIST = [1/8, 1/16, 1/32, 1/64]
H_REF = 1/128

# Manufactured-solution refinement levels.  These are independent of the
# L-shaped reference calculation.
BENCH_H_LIST = [1/8, 1/16, 1/32, 1/64]

# Smooth manufactured solution parameters.
U_SCALE = 1.0e-4


def topology_dict_lshape(nodes, h):
    """Return the three manuscript discretisations on the same nodal set."""
    t3 = np.asarray(fem.generate_triangles(nodes, h), dtype=np.int32)
    q4 = np.asarray(fem.generate_quads(nodes, h), dtype=np.int32)
    ht3, hq4 = fem.generate_hybrid(nodes, h)
    ht3 = np.asarray(ht3, dtype=np.int32)
    hq4 = np.asarray(hq4, dtype=np.int32)
    return {
        "T3": {"T3": t3, "Q4": np.empty((0, 4), dtype=np.int32)},
        "Q4": {"T3": np.empty((0, 3), dtype=np.int32), "Q4": q4},
        "Mixed_T3_Q4": {"T3": ht3, "Q4": hq4},
    }


def topology_counts(elements):
    t3 = np.asarray(elements["T3"])
    q4 = np.asarray(elements["Q4"])
    return len(t3), len(q4), len(t3) + len(q4)


def mesh_dof_rows():
    """Generate the complete mesh/DOF table requested for the manuscript."""
    rows = []

    for h in H_LIST + [H_REF]:
        nodes = np.asarray(fem.generate_nodes(h), dtype=float)
        tops = topology_dict_lshape(nodes, h)

        for topology, elements in tops.items():
            n_t3, n_q4, n_elem = topology_counts(elements)
            n_nodes = len(nodes)
            thermal_dofs = n_nodes
            mechanical_dofs = 2 * n_nodes
            total_sequential = thermal_dofs + mechanical_dofs

            rows.append({
                "topology": topology,
                "h": h,
                "mesh_level": f"1/{int(round(1/h))}",
                "n_nodes": n_nodes,
                "n_T3": n_t3,
                "n_Q4": n_q4,
                "n_elements": n_elem,
                "thermal_DOF": thermal_dofs,
                "mechanical_DOF": mechanical_dofs,
                "total_sequential_unknowns": total_sequential,
                "Q4_fraction_of_elements": n_q4 / n_elem if n_elem else np.nan,
                "domain_area": 0.75,
                "reference_mesh": bool(np.isclose(h, H_REF)),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["h", "topology"], ascending=[False, True]).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Independent manufactured thermoelastic benchmark on Omega=(0,1)^2
# -----------------------------------------------------------------------------
def square_nodes(h):
    n = int(round(1.0 / h))
    x = np.linspace(0.0, 1.0, n + 1)
    y = np.linspace(0.0, 1.0, n + 1)
    nodes = np.array([[xx, yy] for yy in y for xx in x], dtype=float)
    return nodes


def square_index(n):
    return {(i, j): j * (n + 1) + i for j in range(n + 1) for i in range(n + 1)}


def square_t3_q4(h, mixed_pattern="checkerboard"):
    """Create conforming T3, Q4 and mixed T3-Q4 meshes on the unit square."""
    n = int(round(1.0 / h))
    idx = square_index(n)
    t3 = []
    q4 = []

    for j in range(n):
        for i in range(n):
            n1 = idx[(i, j)]
            n2 = idx[(i + 1, j)]
            n3 = idx[(i + 1, j + 1)]
            n4 = idx[(i, j + 1)]
            if mixed_pattern == "checkerboard" and (i + j) % 2 == 0:
                q4.append([n1, n2, n3, n4])
            else:
                t3.append([n1, n2, n3])
                t3.append([n1, n3, n4])

    return (
        np.asarray(t3, dtype=np.int32).reshape(-1, 3),
        np.asarray(q4, dtype=np.int32).reshape(-1, 4),
    )


def square_topologies(h):
    nodes = square_nodes(h)
    n = int(round(1.0 / h))
    idx = square_index(n)

    q4 = []
    t3 = []
    for j in range(n):
        for i in range(n):
            n1 = idx[(i, j)]
            n2 = idx[(i + 1, j)]
            n3 = idx[(i + 1, j + 1)]
            n4 = idx[(i, j + 1)]
            q4.append([n1, n2, n3, n4])
            t3.extend([[n1, n2, n3], [n1, n3, n4]])

    mt3, mq4 = square_t3_q4(h)
    return {
        "T3": {"T3": np.asarray(t3, dtype=np.int32), "Q4": np.empty((0, 4), dtype=np.int32)},
        "Q4": {"T3": np.empty((0, 3), dtype=np.int32), "Q4": np.asarray(q4, dtype=np.int32)},
        "Mixed_T3_Q4": {"T3": mt3, "Q4": mq4},
    }, nodes


def exact_temperature(x, y):
    return 20.0 + 10.0 * x + 5.0 * y


def exact_temperature_array(points):
    p = np.asarray(points)
    return exact_temperature(p[:, 0], p[:, 1])


def exact_displacement(x, y):
    return np.array([
        U_SCALE * (x * x + x * y),
        U_SCALE * (y * y + x * y),
    ], dtype=float)


def exact_strain(x, y):
    return np.array([
        U_SCALE * (2.0 * x + y),
        U_SCALE * (x + 2.0 * y),
        U_SCALE * (x + y),
    ], dtype=float)


def exact_stress(x, y):
    """Exact plane-stress thermoelastic stress in engineering notation."""
    T = exact_temperature(x, y)
    eps = exact_strain(x, y)
    eps_th = fem._thermal_strain(T, fem.ALPHA_TH, fem.T0)
    return fem.elasticity_D(fem.E_MAT, fem.NU_MAT) @ (eps - eps_th)


def manufactured_body_force(x, y):
    """
    Constant body force satisfying -div(sigma*) = f* for the exact field.

    The expression is obtained analytically from the plane-stress constitutive
    law used by the Phase-A core.  The force is per unit volume; the FE load
    assembly multiplies by THICKNESS.
    """
    E = fem.E_MAT
    nu = fem.NU_MAT
    alpha = fem.ALPHA_TH
    s = U_SCALE
    fx = E * (-20.0 * alpha * nu - 20.0 * alpha + nu * s + 5.0 * s) / (2.0 * (nu**2 - 1.0))
    fy = E * (-10.0 * alpha * nu - 10.0 * alpha + nu * s + 5.0 * s) / (2.0 * (nu**2 - 1.0))
    return np.array([fx, fy], dtype=float)


def exact_boundary_conditions(nodes):
    """Full Dirichlet manufactured benchmark: exact T and exact u on boundary."""
    thermal_bc = {}
    mechanical_bc = {}
    tol = 1e-10

    for i, (x, y) in enumerate(nodes):
        if (
            abs(x) < tol or abs(x - 1.0) < tol or
            abs(y) < tol or abs(y - 1.0) < tol
        ):
            thermal_bc[i] = float(exact_temperature(x, y))
            u = exact_displacement(x, y)
            mechanical_bc[2 * i] = float(u[0])
            mechanical_bc[2 * i + 1] = float(u[1])

    return thermal_bc, mechanical_bc


def assemble_manufactured_mechanics(nodes, elements, T, mechanical_bc):
    """Phase-A-compatible mechanical assembly with manufactured body force and BCs."""
    ndof = 2 * len(nodes)
    K = fem.lil_matrix((ndof, ndof))
    F = np.zeros(ndof)
    D = fem.elasticity_D(fem.E_MAT, fem.NU_MAT)
    body = manufactured_body_force(0.0, 0.0)

    def add_element(e, Ke, fth, fbody):
        dofs = np.array([d for n in e for d in (2 * n, 2 * n + 1)], dtype=int)
        for a, I in enumerate(dofs):
            for b, J in enumerate(dofs):
                K[I, J] += Ke[a, b]
            F[I] += fth[a] + fbody[a]

    def t3_body_load(coords):
        # Three-point exact degree-2 triangle rule.
        A = 0.5 * abs(
            (coords[1, 0] - coords[0, 0]) * (coords[2, 1] - coords[0, 1]) -
            (coords[2, 0] - coords[0, 0]) * (coords[1, 1] - coords[0, 1])
        )
        # Barycentric points (1/6,1/6,2/3) and cyclic permutations.
        bary = np.array([
            [1/6, 1/6, 2/3],
            [1/6, 2/3, 1/6],
            [2/3, 1/6, 1/6],
        ])
        f = np.zeros(6)
        for lam in bary:
            p = lam @ coords
            N = lam
            for a in range(3):
                f[2*a:2*a+2] += (A / 3.0) * fem.THICKNESS * N[a] * body
        return f

    def q4_body_load(coords):
        gp = 1.0 / np.sqrt(3.0)
        f = np.zeros(8)
        for xi in (-gp, gp):
            for eta in (-gp, gp):
                N, _, detJ, _ = fem.q4_kinematics(coords, xi, eta)
                for a in range(4):
                    f[2*a:2*a+2] += fem.THICKNESS * N[a] * body * detJ
        return f

    def process_t3(e):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        Ke, _, _ = fem.elasticity_tri_stiffness(
            coords, fem.E_MAT, fem.NU_MAT, fem.THICKNESS
        )
        fth = fem._t3_thermal_load(
            coords, T[e], fem.E_MAT, fem.NU_MAT,
            fem.THICKNESS, fem.ALPHA_TH, fem.T0
        )
        add_element(e, Ke, fth, t3_body_load(coords))

    def process_q4(e):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        Ke, gauss_data, Dloc = fem.elasticity_quad_stiffness(
            coords, fem.E_MAT, fem.NU_MAT, fem.THICKNESS
        )
        fth = fem._q4_thermal_load(
            coords, T[e], gauss_data, Dloc,
            fem.THICKNESS, fem.ALPHA_TH, fem.T0
        )
        add_element(e, Ke, fth, q4_body_load(coords))

    for e in np.asarray(elements["T3"], dtype=int):
        process_t3(e)
    for e in np.asarray(elements["Q4"], dtype=int):
        process_q4(e)

    # Full exact displacement Dirichlet conditions.
    for dof, value in mechanical_bc.items():
        col = np.asarray(K[:, dof].toarray()).ravel()
        F -= col * value
        K[dof, :] = 0.0
        K[:, dof] = 0.0
        K[dof, dof] = 1.0
        F[dof] = value

    return fem.csr_matrix(K), F


def solve_manufactured(h, elements, nodes):
    thermal_bc, mechanical_bc = exact_boundary_conditions(nodes)

    # Thermal problem: exact T is harmonic, so no source term is required.
    Kt, Ft = fem.assemble_thermal(
        nodes, elements, thermal_bc, element_type="Hybrid"
    )
    T = fem.spsolve(Kt, Ft)

    Ke, F = assemble_manufactured_mechanics(
        nodes, elements, T, mechanical_bc
    )
    U = fem.spsolve(Ke, F)

    return T, U


class FEFieldLocator:
    """Locate and evaluate the Phase-A T3/Q4/mixed displacement field."""
    def __init__(self, nodes, elements):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = elements
        self.flat = []
        for e in elements["T3"]:
            self.flat.append(np.asarray(e, dtype=int))
        for e in elements["Q4"]:
            self.flat.append(np.asarray(e, dtype=int))
        self.centers = np.array([self.nodes[e].mean(axis=0) for e in self.flat])
        self.tree = cKDTree(self.centers)

    @staticmethod
    def q4_local(coords, point):
        xi = 0.0
        eta = 0.0
        for _ in range(25):
            N, dxi, deta = fem.q4_shape(xi, eta)
            mapped = np.array([N @ coords[:, 0], N @ coords[:, 1]])
            residual = mapped - point
            if np.linalg.norm(residual) < 1e-12:
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
        if abs(xi) > 1 + 1e-8 or abs(eta) > 1 + 1e-8:
            return None
        return xi, eta

    def evaluate(self, point, U):
        point = np.asarray(point, dtype=float)
        k = min(12, len(self.flat))
        _, candidates = self.tree.query(point, k=k)
        for j in np.atleast_1d(candidates):
            e = self.flat[int(j)]
            coords = self.nodes[e]
            dofs = np.array([d for n in e for d in (2*n, 2*n+1)], dtype=int)
            ue = U[dofs]
            if len(e) == 3:
                N = fem._t3_shape_at_point(coords, point)
                if N is not None:
                    return N @ ue.reshape(-1, 2)
            else:
                N = fem._q4_shape_at_point(coords, point)
                if N is not None:
                    return N @ ue.reshape(-1, 2)
        # Robust boundary fallback.
        for e in self.flat:
            coords = self.nodes[e]
            dofs = np.array([d for n in e for d in (2*n, 2*n+1)], dtype=int)
            ue = U[dofs]
            if len(e) == 3:
                N = fem._t3_shape_at_point(coords, point)
                if N is not None:
                    return N @ ue.reshape(-1, 2)
            else:
                N = fem._q4_shape_at_point(coords, point)
                if N is not None:
                    return N @ ue.reshape(-1, 2)
        raise RuntimeError(f"Could not locate point {point}")


def element_quadrature(coords):
    """Return physical quadrature points, weights, shape functions and local coordinates."""
    if len(coords) == 3:
        A = 0.5 * abs(
            (coords[1, 0] - coords[0, 0]) * (coords[2, 1] - coords[0, 1]) -
            (coords[2, 0] - coords[0, 0]) * (coords[1, 1] - coords[0, 1])
        )
        bary = np.array([
            [1/6, 1/6, 2/3],
            [1/6, 2/3, 1/6],
            [2/3, 1/6, 1/6],
        ])
        return [(lam @ coords, A / 3.0, lam, None) for lam in bary]

    gp = 1.0 / np.sqrt(3.0)
    out = []
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            N, _, detJ, _ = fem.q4_kinematics(coords, xi, eta)
            out.append((N @ coords, detJ, N, (xi, eta)))
    return out


def benchmark_errors(nodes, elements, T, U):
    """Compute exact-field L2 errors directly at element quadrature points."""
    D = fem.elasticity_D(fem.E_MAT, fem.NU_MAT)

    accum = {
        "temperature_L2_error": 0.0,
        "displacement_L2_error": 0.0,
        "engineering_strain_L2_error": 0.0,
        "stress_L2_error": 0.0,
        "temperature_norm": 0.0,
        "displacement_norm": 0.0,
        "strain_norm": 0.0,
        "stress_norm": 0.0,
    }

    for e in list(elements["T3"]) + list(elements["Q4"]):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        dofs = np.array([d for n in e for d in (2*n, 2*n+1)], dtype=int)
        ue = U[dofs]

        for p, w, N, local in element_quadrature(coords):
            x, y = p
            T_h = float(N @ T[e])
            u_h = N @ ue.reshape(-1, 2)
            T_ex = float(exact_temperature(x, y))
            u_ex = exact_displacement(x, y)
            eps_ex = exact_strain(x, y)
            sig_ex = exact_stress(x, y)

            if len(e) == 3:
                _, B, _ = fem.elasticity_tri_stiffness(
                    coords, fem.E_MAT, fem.NU_MAT, fem.THICKNESS
                )
            else:
                xi, eta = local
                _, B, _, _ = fem.q4_kinematics(coords, xi, eta)

            eps_h = B @ ue
            eps_th_h = fem._thermal_strain(T_h, fem.ALPHA_TH, fem.T0)
            sig_h = D @ (eps_h - eps_th_h)

            dt = T_h - T_ex
            du = u_h - u_ex
            de = eps_h - eps_ex
            ds = sig_h - sig_ex

            accum["temperature_L2_error"] += w * dt**2
            accum["displacement_L2_error"] += w * float(du @ du)
            accum["engineering_strain_L2_error"] += w * float(de @ de)
            accum["stress_L2_error"] += w * float(ds @ ds)
            accum["temperature_norm"] += w * T_ex**2
            accum["displacement_norm"] += w * float(u_ex @ u_ex)
            accum["strain_norm"] += w * float(eps_ex @ eps_ex)
            accum["stress_norm"] += w * float(sig_ex @ sig_ex)

    out = {
        "temperature_L2_error": np.sqrt(max(accum["temperature_L2_error"], 0.0)),
        "displacement_L2_error": np.sqrt(max(accum["displacement_L2_error"], 0.0)),
        "engineering_strain_L2_error": np.sqrt(max(accum["engineering_strain_L2_error"], 0.0)),
        "stress_L2_error": np.sqrt(max(accum["stress_L2_error"], 0.0)),
    }
    out["temperature_relative_L2_error"] = out["temperature_L2_error"] / np.sqrt(max(accum["temperature_norm"], 1e-300))
    out["displacement_relative_L2_error"] = out["displacement_L2_error"] / np.sqrt(max(accum["displacement_norm"], 1e-300))
    out["engineering_strain_relative_error"] = out["engineering_strain_L2_error"] / np.sqrt(max(accum["strain_norm"], 1e-300))
    out["stress_relative_L2_error"] = out["stress_L2_error"] / np.sqrt(max(accum["stress_norm"], 1e-300))
    return out

def observed_orders(df, error_columns):
    df = df.copy()
    for topology in df["topology"].unique():
        idx = df.index[df["topology"] == topology].tolist()
        idx = sorted(idx, key=lambda i: df.loc[i, "h"], reverse=True)
        for err, order in error_columns.items():
            df.loc[idx[0], order] = np.nan
            for k in range(1, len(idx)):
                e0 = float(df.loc[idx[k-1], err])
                e1 = float(df.loc[idx[k], err])
                h0 = float(df.loc[idx[k-1], "h"])
                h1 = float(df.loc[idx[k], "h"])
                if e0 > 0 and e1 > 0:
                    df.loc[idx[k], order] = np.log(e0/e1) / np.log(h0/h1)
    return df


def run_lshape_mesh_table():
    df = mesh_dof_rows()
    full_file = OUT_DIR / "lshape_mesh_dof_table.csv"
    df.to_csv(full_file, index=False, float_format="%.10e")

    # Manuscript-oriented version: one row per topology/refinement level.
    manuscript_cols = [
        "topology", "mesh_level", "h", "n_nodes", "n_T3", "n_Q4",
        "n_elements", "thermal_DOF", "mechanical_DOF",
        "total_sequential_unknowns"
    ]
    df[manuscript_cols].to_csv(
        OUT_DIR / "lshape_mesh_dof_table_manuscript.csv",
        index=False,
        float_format="%.10e",
    )
    return df


def run_manufactured_verification():
    rows = []
    for h in BENCH_H_LIST:
        topologies, nodes = square_topologies(h)
        print(f"\nManufactured benchmark h={h:g}, nodes={len(nodes)}")

        for topology, elements in topologies.items():
            n_t3, n_q4, n_elem = topology_counts(elements)
            n_nodes = len(nodes)
            start = time.perf_counter()
            T, U = solve_manufactured(h, elements, nodes)
            elapsed = time.perf_counter() - start

            errors = benchmark_errors(nodes, elements, T, U)
            rows.append({
                "topology": topology,
                "h": h,
                "mesh_level": f"1/{int(round(1/h))}",
                "n_nodes": n_nodes,
                "n_T3": n_t3,
                "n_Q4": n_q4,
                "n_elements": n_elem,
                "thermal_DOF": n_nodes,
                "mechanical_DOF": 2 * n_nodes,
                "total_sequential_unknowns": 3 * n_nodes,
                "wall_time_s": elapsed,
                **errors,
            })

            print(
                f"  {topology:12s}: T3={n_t3:5d}, Q4={n_q4:5d}, "
                f"T-L2={errors['temperature_L2_error']:.3e}, "
                f"u-L2={errors['displacement_L2_error']:.3e}, "
                f"eps-L2={errors['engineering_strain_L2_error']:.3e}, "
                f"sig-L2={errors['stress_L2_error']:.3e}"
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["topology", "h"], ascending=[True, False]).reset_index(drop=True)
    df = observed_orders(df, {
        "temperature_L2_error": "temperature_order",
        "displacement_L2_error": "displacement_order",
        "engineering_strain_L2_error": "strain_order",
        "stress_L2_error": "stress_order",
    })

    df.to_csv(
        OUT_DIR / "manufactured_verification_results.csv",
        index=False,
        float_format="%.10e",
    )

    # Compact manuscript table.
    summary_cols = [
        "topology", "mesh_level", "h", "n_nodes", "n_T3", "n_Q4",
        "thermal_DOF", "mechanical_DOF", "total_sequential_unknowns",
        "temperature_L2_error", "temperature_order",
        "displacement_L2_error", "displacement_order",
        "engineering_strain_L2_error", "strain_order",
        "stress_L2_error", "stress_order", "wall_time_s",
    ]
    df[summary_cols].to_csv(
        OUT_DIR / "manufactured_verification_summary.csv",
        index=False,
        float_format="%.10e",
    )

    inputs = pd.DataFrame([
        ["domain", "Omega_b", "(0,1)^2"],
        ["temperature", "T*(x,y)", "20 + 10*x + 5*y"],
        ["displacement", "ux*(x,y)", f"{U_SCALE:g}*(x^2 + x*y)"],
        ["displacement", "uy*(x,y)", f"{U_SCALE:g}*(y^2 + x*y)"],
        ["thermal_source", "q", "0"],
        ["mechanical_boundary", "Dirichlet", "exact u* on entire boundary"],
        ["thermal_boundary", "Dirichlet", "exact T* on entire boundary"],
        ["body_force", "f*", "-div(sigma*); analytic constant vector"],
        ["element_quadrature", "T3", "3-point triangle rule for verification integrals/body load"],
        ["element_quadrature", "Q4", "2x2 Gauss rule"],
        ["plane_state", "constitutive law", "plane stress"],
    ], columns=["category", "quantity", "definition"])
    inputs.to_csv(OUT_DIR / "manufactured_verification_inputs.csv", index=False)
    return df


def main():
    print("=" * 82)
    print("COMPLETE MESH/DOF TABLE + INDEPENDENT VERIFICATION")
    print("=" * 82)
    print(f"Validation core : {CORE.resolve()}")
    print(f"Output dir   : {OUT_DIR.resolve()}")

    mesh_df = run_lshape_mesh_table()
    print("\nL-shaped mesh/DOF table:") #Table 3: Mesh characteristics and algebraic problem size for the three discretisations
    print(mesh_df.to_string(index=False))

    verification_df = run_manufactured_verification()
    print("\nManufactured-solution verification results:") #See Table 2: Independent manufactured-solution verification for the T3, Q4 and mixed T3–Q4 discretisations.
    print(verification_df.to_string(index=False))

    print("\nSaved files:")
    for p in sorted(OUT_DIR.glob("*.csv")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
