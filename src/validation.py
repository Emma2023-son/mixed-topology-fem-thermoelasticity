import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as mtri
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import time
import pandas as pd
import os
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# Set matplotlib parameters for better visualization
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'figure.dpi': 150
})

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent

RESULT_DIR = BASE_DIR / "validation_results"
FIG_DIR = BASE_DIR / "validation_figures"

RESULT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ============================================================
# PHASE A — CORRECTED FEM CORE
# ============================================================
# This section intentionally overrides the earlier development versions.
# It is the authoritative implementation for the Phase-A audit.

from scipy.spatial import cKDTree

T0 = 20.0
E_MAT = 70e9
NU_MAT = 0.33
THICKNESS = 0.01
ALPHA_TH = 1.0e-5
K_TH = 1.0

# ----------------------------
# 1. Domain and Mesh Generation
# ----------------------------
def standard_l_shape_domain(x, y):
    """Standard L-shape domain (unit square with [0.5,1]x[0.5,1] removed)"""
    return (x <= 0.5 or y <= 0.5) and 0 <= x <= 1 and 0 <= y <= 1

def generate_nodes(h, domain_func=standard_l_shape_domain):
    """Generate nodes with improved spacing and boundary handling"""
    nodes = []
    # Add boundary nodes first to ensure proper connectivity
    for x in np.arange(0, 1 + 1e-9, h):
        if domain_func(x, 0):
            nodes.append([x, 0])
        if domain_func(x, 1):
            nodes.append([x, 1])
    
    for y in np.arange(0, 1 + 1e-9, h):
        if domain_func(0, y):
            nodes.append([0, y])
        if domain_func(1, y):
            nodes.append([1, y])
    
    # Add internal nodes
    for y in np.arange(h, 1, h):
        for x in np.arange(h, 1, h):
            if domain_func(x, y):
                nodes.append([x, y])
    
    # Remove duplicates and sort
    nodes = np.array(list(set(tuple(node) for node in nodes)))
    nodes = nodes[np.lexsort((nodes[:,1], nodes[:,0]))]
    
    return np.array(nodes)

def generate_triangles(nodes, h):
    """Generate triangular mesh using Delaunay triangulation for better quality"""
    try:
        tri = Delaunay(nodes)
        triangles = tri.simplices
        
        # Filter out elements outside the domain
        valid_triangles = []
        for t in triangles:
            centroid = np.mean(nodes[t], axis=0)
            if standard_l_shape_domain(centroid[0], centroid[1]):
                valid_triangles.append(t)
        
        return np.array(valid_triangles)
    except:
        # Fallback to structured mesh if Delaunay fails
        idx = {(round(x/h), round(y/h)): i for i,(x,y) in enumerate(nodes)}
        elements = []
        for j in range(int(1/h)):
            for i in range(int(1/h)):
                if (i,j) in idx and (i+1,j) in idx and (i,j+1) in idx and (i+1,j+1) in idx:
                    elements.append([idx[(i,j)], idx[(i+1,j)], idx[(i+1,j+1)]])
                    elements.append([idx[(i,j)], idx[(i+1,j+1)], idx[(i,j+1)]])
        return np.array(elements)



def generate_quads(nodes, h):
    """Generate quadrilateral mesh with improved boundary handling"""
    idx = {(round(x/h), round(y/h)): i for i,(x,y) in enumerate(nodes)}
    elements = []
    
    for j in range(int(1/h)):
        for i in range(int(1/h)):
            if (i,j) in idx and (i+1,j) in idx and (i,j+1) in idx and (i+1,j+1) in idx:
                # Check if the quadrilateral is entirely within the domain
                pts = [nodes[idx[(i,j)]], nodes[idx[(i+1,j)]], 
                       nodes[idx[(i+1,j+1)]], nodes[idx[(i,j+1)]]]
                centroid = np.mean(pts, axis=0)
                if standard_l_shape_domain(centroid[0], centroid[1]):
                    elements.append([idx[(i,j)], idx[(i+1,j)], 
                                    idx[(i+1,j+1)], idx[(i,j+1)]])
    
    return np.array(elements)

def generate_hybrid(nodes, h, transition_radius=0.22):
    """
    Generate a genuinely heterogeneous hybrid mesh.

    Quadrilateral elements are retained over most of the domain,
    while cells within a prescribed radius of the re-entrant corner
    are subdivided into triangular elements.

    Parameters
    ----------
    nodes : ndarray of shape (N,2)
        Mesh node coordinates.
    h : float
        Structured mesh spacing.
    transition_radius : float, optional
        Radius around the re-entrant corner where quadrilateral
        elements are converted into triangular elements.

    Returns
    -------
    triangles : ndarray of shape (NT,3)
        Connectivity of triangular elements.

    quads : ndarray of shape (NQ,4)
        Connectivity of quadrilateral elements.
    """

    # ------------------------------------------------------------
    # Build node lookup table
    # ------------------------------------------------------------
    idx = {
        (int(round(x / h)), int(round(y / h))): k
        for k, (x, y) in enumerate(nodes)
    }

    triangles = []
    quads = []

    nx = int(round(1.0 / h))
    ny = int(round(1.0 / h))

    # Re-entrant corner of the L-shaped domain
    corner = np.array([0.5, 0.5])

    for j in range(ny):
        for i in range(nx):

            keys = [
                (i, j),
                (i + 1, j),
                (i + 1, j + 1),
                (i, j + 1)
            ]

            # Skip incomplete cells
            if not all(key in idx for key in keys):
                continue

            n1 = idx[(i, j)]
            n2 = idx[(i + 1, j)]
            n3 = idx[(i + 1, j + 1)]
            n4 = idx[(i, j + 1)]

            cell_nodes = np.array([n1, n2, n3, n4])

            centroid = nodes[cell_nodes].mean(axis=0)

            # Cell must lie inside the L-shaped domain
            if not standard_l_shape_domain(centroid[0], centroid[1]):
                continue

            # Distance from the re-entrant corner
            r = np.linalg.norm(centroid - corner)

            if r <= transition_radius:

                # Alternate the diagonal orientation
                if (i + j) % 2 == 0:
                    triangles.append([n1, n2, n3])
                    triangles.append([n1, n3, n4])
                else:
                    triangles.append([n1, n2, n4])
                    triangles.append([n2, n3, n4])

            else:

                quads.append([n1, n2, n3, n4])

    # ------------------------------------------------------------
    # Convert to NumPy arrays
    # ------------------------------------------------------------
    triangles = np.asarray(triangles, dtype=np.int32)
    quads = np.asarray(quads, dtype=np.int32)

    # Ensure correct shapes even if empty
    if triangles.size == 0:
        triangles = np.empty((0, 3), dtype=np.int32)
    else:
        triangles = triangles.reshape(-1, 3)

    if quads.size == 0:
        quads = np.empty((0, 4), dtype=np.int32)
    else:
        quads = quads.reshape(-1, 4)

    return triangles, quads

def elements_to_triangles(elements):
    """
    Convert T3 / Q4 / Hybrid elements into pure triangles for plotting.
    """
    triangles = []

    if isinstance(elements, dict):  # Hybrid
        # T3 elements
        for e in elements["T3"]:
            triangles.append(e)

        # Q4 → split into two triangles
        for e in elements["Q4"]:
            triangles.append([e[0], e[1], e[2]])
            triangles.append([e[0], e[2], e[3]])

    else:
        for e in elements:
            if len(e) == 3:          # T3
                triangles.append(e)
            elif len(e) == 4:        # Q4
                triangles.append([e[0], e[1], e[2]])
                triangles.append([e[0], e[2], e[3]])

    return np.array(triangles, dtype=int)

# ----------------------------
# 2. Thermal FEM Assembly
# ----------------------------
def thermal_tri_stiffness(coords, k=1.0):
    """Improved stiffness calculation with better numerical stability"""
    n = len(coords)
    if n == 3:  # Triangle
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        
        # Calculate area using a more stable formula
        A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if A < 1e-12:
            return np.zeros((3, 3)), A
        
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        Ke = k * (np.outer(b, b) + np.outer(c, c)) / (4 * A)
        return Ke, A
    elif n == 4:  # Quadrilateral (bilinear Q4)
        x = coords[:, 0]
        y = coords[:, 1]
        Ke = np.zeros((4, 4))
        
        # 2x2 Gauss quadrature with improved numerical stability
        gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
        for xi in gp:
            for eta in gp:
                dN_dxi = np.array([
                    -0.25 * (1 - eta), 0.25 * (1 - eta), 
                    0.25 * (1 + eta), -0.25 * (1 + eta)
                ])
                dN_deta = np.array([
                    -0.25 * (1 - xi), -0.25 * (1 + xi),
                    0.25 * (1 + xi), 0.25 * (1 - xi)
                ])
                
                J = np.array([
                    [np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
                    [np.dot(dN_deta, x), np.dot(dN_deta, y)]
                ])
                
                detJ = np.linalg.det(J)
                if abs(detJ) < 1e-12:
                    continue
                    
                invJ = np.linalg.inv(J)
                dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
                dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
                
                B = np.vstack([dN_dx, dN_dy])
                Ke += k * (B.T @ B) * detJ
        return Ke, None
    else:
        raise ValueError("Unsupported element type")

# ----------------------------
# 3. Elasticity FEM Assembly
# ----------------------------
def elasticity_tri_stiffness(coords, E, nu, t):
    """Improved elasticity stiffness calculation"""
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    
    # Calculate area with better numerical stability
    A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if A < 1e-12:
        return np.zeros((6, 6)), None, None
    
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])
    
    # Strain-displacement matrix
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2*i] = b[i]
        B[1, 2*i+1] = c[i]
        B[2, 2*i] = c[i]
        B[2, 2*i+1] = b[i]
    B /= (2 * A)
    
    # Constitutive matrix
    D = E / (1 - nu**2) * np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1 - nu) / 2]
    ])
    
    Ke = t * A * B.T @ D @ B
    return Ke, B, D

def elasticity_D(E=E_MAT, nu=NU_MAT):
    """Plane-stress constitutive matrix in engineering shear notation."""
    return E / (1.0 - nu**2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])


def q4_shape(xi, eta):
    N = 0.25 * np.array([
        (1-xi)*(1-eta),
        (1+xi)*(1-eta),
        (1+xi)*(1+eta),
        (1-xi)*(1+eta)
    ])
    dN_dxi = 0.25 * np.array([
        -(1-eta), (1-eta), (1+eta), -(1+eta)
    ])
    dN_deta = 0.25 * np.array([
        -(1-xi), -(1+xi), (1+xi), (1-xi)
    ])
    return N, dN_dxi, dN_deta


def q4_kinematics(coords, xi, eta):
    """Return Q4 shape functions, B, detJ and J at one Gauss point."""
    N, dN_dxi, dN_deta = q4_shape(xi, eta)
    x = coords[:, 0]
    y = coords[:, 1]
    J = np.array([
        [np.dot(dN_dxi, x), np.dot(dN_deta, x)],
        [np.dot(dN_dxi, y), np.dot(dN_deta, y)]
    ])
    detJ = np.linalg.det(J)
    if detJ <= 1e-12:
        raise ValueError(f"Invalid Q4 Jacobian determinant: {detJ}")
    invJ = np.linalg.inv(J)
    grad = invJ @ np.vstack([dN_dxi, dN_deta])
    dN_dx = grad[0]
    dN_dy = grad[1]

    B = np.zeros((3, 8))
    for a in range(4):
        B[0, 2*a] = dN_dx[a]
        B[1, 2*a+1] = dN_dy[a]
        B[2, 2*a] = dN_dy[a]
        B[2, 2*a+1] = dN_dx[a]
    return N, B, detJ, J


def elasticity_quad_stiffness(coords, E=E_MAT, nu=NU_MAT, t=THICKNESS):
    """Consistent 2x2 Gauss integration for Q4 plane-stress elasticity."""
    D = elasticity_D(E, nu)
    gp = 1.0 / np.sqrt(3.0)
    Ke = np.zeros((8, 8))
    gauss_data = []
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            N, B, detJ, J = q4_kinematics(coords, xi, eta)
            Ke += t * (B.T @ D @ B) * detJ
            gauss_data.append({
                'xi': xi, 'eta': eta, 'N': N, 'B': B,
                'detJ': detJ, 'J': J
            })
    return Ke, gauss_data, D


def thermal_q4_stiffness(coords, k=K_TH):
    """Consistent 2x2 Gauss integration for Q4 heat conduction."""
    gp = 1.0 / np.sqrt(3.0)
    Ke = np.zeros((4, 4))
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            N, _, detJ, _ = q4_kinematics(coords, xi, eta)
            _, dN_dxi, dN_deta = q4_shape(xi, eta)
            J = np.array([
                [np.dot(dN_dxi, coords[:,0]), np.dot(dN_deta, coords[:,0])],
                [np.dot(dN_dxi, coords[:,1]), np.dot(dN_deta, coords[:,1])]
            ])
            grad = np.linalg.inv(J) @ np.vstack([dN_dxi, dN_deta])
            G = grad
            Ke += k * (G.T @ G) * detJ
    return Ke

def von_mises(sigma):
    """Improved von Mises calculation with numerical stability"""
    sx, sy, txy = sigma
    return np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)


def assemble_thermal(nodes, elements, bc, element_type="T3"):
    """Assemble scalar thermal diffusion system for T3, Q4 or Hybrid."""
    n = len(nodes)
    K = lil_matrix((n, n))
    F = np.zeros(n)

    def add(e, Ke):
        for a, I in enumerate(e):
            for b, J in enumerate(e):
                K[I, J] += Ke[a, b]

    if element_type == "T3":
        for e in elements:
            e = np.asarray(e, dtype=int)
            Ke, _ = thermal_tri_stiffness(nodes[e], k=K_TH)
            add(e, Ke)
    elif element_type == "Q4":
        for e in elements:
            e = np.asarray(e, dtype=int)
            add(e, thermal_q4_stiffness(nodes[e], k=K_TH))
    elif element_type == "Hybrid":
        for e in elements["T3"]:
            e = np.asarray(e, dtype=int)
            Ke, _ = thermal_tri_stiffness(nodes[e], k=K_TH)
            add(e, Ke)
        for e in elements["Q4"]:
            e = np.asarray(e, dtype=int)
            add(e, thermal_q4_stiffness(nodes[e], k=K_TH))
    else:
        raise ValueError(f"Unknown element_type={element_type}")

    # Strong Dirichlet enforcement with correct RHS elimination.
    # For a prescribed value T_i=g_i, the eliminated column contributes
    # -K[:,i] g_i to the remaining right-hand side.
    for node, val in bc.items():
        col = np.asarray(K[:, node].toarray()).ravel()
        F -= col * val
        K[node, :] = 0.0
        K[:, node] = 0.0
        K[node, node] = 1.0
        F[node] = val
    return csr_matrix(K), F


def _thermal_strain(Tgp, alpha=ALPHA_TH, T0_=T0):
    return alpha * (Tgp - T0_) * np.array([1.0, 1.0, 0.0])


def _t3_thermal_load(coords, T_e, E, nu, t, alpha, T0_):
    """Exact one-point/constant-strain thermal load for linear T3."""
    A = 0.5 * abs(
        (coords[1,0]-coords[0,0])*(coords[2,1]-coords[0,1]) -
        (coords[2,0]-coords[0,0])*(coords[1,1]-coords[0,1])
    )
    _, B, D = elasticity_tri_stiffness(coords, E, nu, t)
    T_centroid = np.mean(T_e)
    eps_th = _thermal_strain(T_centroid, alpha, T0_)
    return t * A * (B.T @ D @ eps_th)

def _flatten_elements(elements, element_type):
    if element_type == "Hybrid":
        return [np.asarray(e, dtype=int) for e in elements["T3"]] + [np.asarray(e, dtype=int) for e in elements["Q4"]]
    return [np.asarray(e, dtype=int) for e in elements]

def _t3_shape_at_point(coords, p):
    x1, y1 = coords[0]; x2, y2 = coords[1]; x3, y3 = coords[2]
    det = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    if abs(det) < 1e-14: return None
    N1 = ((x2-p[0])*(y3-p[1]) - (x3-p[0])*(y2-p[1])) / det
    N2 = ((x3-p[0])*(y1-p[1]) - (x1-p[0])*(y3-p[1])) / det
    N3 = 1.0 - N1 - N2
    N = np.array([N1, N2, N3])
    if np.min(N) < -1e-9 or np.max(N) > 1+1e-9: return None
    return N

def _q4_shape_at_point(coords, p, tol=1e-8):
    xi = eta = 0.0
    for _ in range(12):
        N, dN_dxi, dN_deta = q4_shape(xi, eta)
        xmap = np.dot(N, coords[:,0]); ymap = np.dot(N, coords[:,1])
        res = np.array([xmap-p[0], ymap-p[1]])
        if np.linalg.norm(res) < 1e-11: break
        J = np.array([
            [np.dot(dN_dxi, coords[:,0]), np.dot(dN_deta, coords[:,0])],
            [np.dot(dN_dxi, coords[:,1]), np.dot(dN_deta, coords[:,1])]
        ])
        delta = np.linalg.solve(J, res)
        xi -= delta[0]; eta -= delta[1]
    if abs(xi) > 1+tol or abs(eta) > 1+tol:
        return None
    return q4_shape(xi, eta)[0]


def _target_qps(coords):
    """Return physical quadrature points, weights, N and B for one target element."""
    if len(coords) == 3:
        A = 0.5 * abs(
            (coords[1,0]-coords[0,0])*(coords[2,1]-coords[0,1]) -
            (coords[2,0]-coords[0,0])*(coords[1,1]-coords[0,1])
        )
        _, B, _ = elasticity_tri_stiffness(coords, E_MAT, NU_MAT, THICKNESS)
        # Linear shape functions evaluated at centroid.
        N = np.ones(3)/3.0
        p = N @ coords
        return [(p, A, N, B)]
    gp = 1.0/np.sqrt(3.0)
    out = []
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            N, B, detJ, _ = q4_kinematics(coords, xi, eta)
            p = N @ coords
            out.append((p, detJ, N, B))
    return out

def _q4_thermal_load(coords, T_e, gauss_data, D, t, alpha, T0_):
    """Consistent 2x2 Gauss integration of Q4 thermal load."""
    fth = np.zeros(8)
    for g in gauss_data:
        Tgp = np.dot(g['N'], T_e)
        eps_th = _thermal_strain(Tgp, alpha, T0_)
        fth += t * (g['B'].T @ D @ eps_th) * g['detJ']
    return fth


def assemble_elasticity(nodes, elements, E=E_MAT, nu=NU_MAT,
                        t=THICKNESS, T=None, alpha=ALPHA_TH,
                        element_type="T3"):
    """Assemble plane-stress thermoelasticity with consistent thermal loads."""
    ndof = 2 * len(nodes)
    K = lil_matrix((ndof, ndof))
    F = np.zeros(ndof)
    D = elasticity_D(E, nu)

    def add_element(e, Ke, fth):
        dofs = np.array([d for n in e for d in (2*n, 2*n+1)], dtype=int)
        for a, I in enumerate(dofs):
            for b, J in enumerate(dofs):
                K[I, J] += Ke[a, b]
            F[I] += fth[a]

    def process_t3(e):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        Ke, B, Dloc = elasticity_tri_stiffness(coords, E, nu, t)
        fth = np.zeros(6)
        if T is not None:
            fth = _t3_thermal_load(coords, T[e], E, nu, t, alpha, T0)
        add_element(e, Ke, fth)

    def process_q4(e):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        Ke, gauss_data, Dloc = elasticity_quad_stiffness(coords, E, nu, t)
        fth = np.zeros(8)
        if T is not None:
            fth = _q4_thermal_load(coords, T[e], gauss_data, Dloc, t, alpha, T0)
        add_element(e, Ke, fth)

    if element_type == "T3":
        for e in elements: process_t3(e)
    elif element_type == "Q4":
        for e in elements: process_q4(e)
    elif element_type == "Hybrid":
        for e in elements["T3"]: process_t3(e)
        for e in elements["Q4"]: process_q4(e)
    else:
        raise ValueError(f"Unknown element_type={element_type}")

    # Mechanical supports used by the current benchmark:
    # ux = 0 on x=0; uy = 0 on y=0. This removes rigid-body modes
    # while leaving the re-entrant corner faces traction-free.
    for i, (x, y) in enumerate(nodes):
        if abs(x) < 1e-9:
            d = 2*i
            K[d, :] = 0.0; K[:, d] = 0.0; K[d, d] = 1.0; F[d] = 0.0
        if abs(y) < 1e-9:
            d = 2*i + 1
            K[d, :] = 0.0; K[:, d] = 0.0; K[d, d] = 1.0; F[d] = 0.0

    return csr_matrix(K), F


def compute_stress(nodes, elements, U, E=E_MAT, nu=NU_MAT,
                   t=THICKNESS, element_type="T3", T=None,
                   alpha=ALPHA_TH, T0_=T0):
    """Return element-averaged thermoelastic stress and von Mises stress.

    T3 stress is constant. Q4 stress is evaluated at all four Gauss points
    and averaged with the physical Gauss weights. Thermal strain is included.
    """
    stresses, vmises, centroids = [], [], []
    D = elasticity_D(E, nu)

    def t3_stress(e):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        _, B, _ = elasticity_tri_stiffness(coords, E, nu, t)
        dofs = np.array([d for n in e for d in (2*n, 2*n+1)], dtype=int)
        ue = U[dofs]
        eps = B @ ue
        if T is not None:
            eps -= _thermal_strain(np.mean(T[e]), alpha, T0_)
        sigma = D @ eps
        return sigma, nodes[e].mean(axis=0)

    def q4_stress(e):
        e = np.asarray(e, dtype=int)
        coords = nodes[e]
        _, gauss_data, Dloc = elasticity_quad_stiffness(coords, E, nu, t)
        dofs = np.array([d for n in e for d in (2*n, 2*n+1)], dtype=int)
        ue = U[dofs]
        sigma_sum = np.zeros(3)
        weight_sum = 0.0
        for g in gauss_data:
            eps = g['B'] @ ue
            if T is not None:
                Tgp = np.dot(g['N'], T[e])
                eps -= _thermal_strain(Tgp, alpha, T0_)
            sigma = Dloc @ eps
            sigma_sum += sigma * g['detJ']
            weight_sum += g['detJ']
        return sigma_sum / weight_sum, coords.mean(axis=0)

    if element_type in ("T3", "Q4"):
        iterator = elements
        for e in iterator:
            sigma, c = t3_stress(e) if element_type == "T3" else q4_stress(e)
            stresses.append(sigma); centroids.append(c); vmises.append(von_mises(sigma))
    elif element_type == "Hybrid":
        for e in elements["T3"]:
            sigma, c = t3_stress(e)
            stresses.append(sigma); centroids.append(c); vmises.append(von_mises(sigma))
        for e in elements["Q4"]:
            sigma, c = q4_stress(e)
            stresses.append(sigma); centroids.append(c); vmises.append(von_mises(sigma))
    else:
        raise ValueError(f"Unknown element_type={element_type}")

    return np.asarray(stresses), np.asarray(vmises), np.asarray(centroids)



def solve_thermoelastic(h, element_type="T3", nodes=None, elements=None):
    """Authoritative Phase-A thermoelastic solver."""
    if nodes is None or elements is None:
        nodes = generate_nodes(h)
        if element_type == "T3":
            elements = generate_triangles(nodes, h)
        elif element_type == "Q4":
            elements = generate_quads(nodes, h)
        elif element_type == "Hybrid":
            t3, q4 = generate_hybrid(nodes, h)
            elements = {"T3": t3, "Q4": q4}
        else:
            raise ValueError(element_type)

    # Thermal Dirichlet conditions in the current benchmark.
    
    bc = {}

    for i, (x, y) in enumerate(nodes):

        # Left boundary: T = 20
        if abs(x) < 1e-9:
            bc[i] = 20.0

        # Bottom boundary: T = 50
        if abs(y) < 1e-9:
            bc[i] = 50.0

        # Right boundary: T = 100
        # Includes the corner (1,0) by convention.
        if abs(x - 1.0) < 1e-9 and y <= 0.5 + 1e-9:
            bc[i] = 100.0
    
    Kt, Ft = assemble_thermal(nodes, elements, bc, element_type)
    T = spsolve(Kt, Ft)

    Ke, F = assemble_elasticity(
        nodes, elements, E_MAT, NU_MAT, THICKNESS,
        T=T, alpha=ALPHA_TH, element_type=element_type
    )
    U = spsolve(Ke, F)
    disp = np.hypot(U[0::2], U[1::2])
    stresses, vmises, centroids = compute_stress(
        nodes, elements, U, E_MAT, NU_MAT, THICKNESS,
        element_type=element_type, T=T, alpha=ALPHA_TH, T0_=T0
    )
    return nodes, elements, T, U, disp, stresses, vmises, centroids

def plot_mesh(nodes, elements, title="Mesh", fname=None):
    """Improved mesh plotting with better visualization"""
    plt.figure(figsize=(8, 6))
    tri_elements = elements_to_triangles(elements)
    
    for e in tri_elements:
        pts = nodes[e]
        plt.fill(pts[:,0], pts[:,1], edgecolor='k', fill=False, linewidth=0.5)
    
    plt.scatter(nodes[:,0], nodes[:,1], c='r', s=10)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    
    plt.close()

def plot_field(nodes, elements, field, title="Field",
               cmap="viridis", fname=None, levels=30):
    """Improved field plotting with better normalization and labels"""
    plt.figure(figsize=(8, 6))
    
    tri_elements = elements_to_triangles(elements)
    tri = mtri.Triangulation(
        nodes[:,0],
        nodes[:,1],
        tri_elements
    )
    
    # Normalize field for better visualization
    field_min, field_max = np.min(field), np.max(field)
    if field_max - field_min > 1e-10:
        field_norm = (field - field_min) / (field_max - field_min)
    else:
        field_norm = field
    
    tcf = plt.tricontourf(tri, field, levels=levels, cmap=cmap)
    cbar = plt.colorbar(tcf)
    cbar.set_label(title)
    
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()



def phase_a_validation():
    """Minimal validation driver; no paper FIG_DIR/results are generated."""
    print("="*72)
    print("PHASE A — FEM IMPLEMENTATION VALIDATION")
    print("="*72)
    for etype in ("T3", "Q4", "Hybrid"):
        print(f"\n--- {etype} ---")
        for h in (0.25, 0.125):
            nodes, elements, T, U, disp, stresses, vmises, centroids = solve_thermoelastic(h, etype)
            print(f"h={h:g}: nodes={len(nodes)}, elements=" +
                  (f"{len(elements['T3'])+len(elements['Q4'])} (T3={len(elements['T3'])}, Q4={len(elements['Q4'])})" if etype=="Hybrid" else str(len(elements))))
            print(f"  T range = [{T.min():.6g}, {T.max():.6g}]")
            print(f"  max |u| = {disp.max():.6e}")
            print(f"  max VM  = {vmises.max():.6e}")
            
            # Mesh & Field Plots
            plot_mesh(nodes, elements,
                        title=f"{etype} Mesh",
                        fname=FIG_DIR/f"mesh_{etype}.png")
            
            
            plot_field(nodes, elements, T,
                               title=f"Temperature ({etype})",
                               fname=FIG_DIR/f"temperature_{etype}.png")
                    
            plot_field(nodes, elements, disp,
                        title=f"Displacement ({etype})",
                        fname=FIG_DIR/f"displacement_{etype}.png")
            
    print("\nPHASE A CORE VALIDATION COMPLETED.")


if __name__ == "__main__":
    phase_a_validation()
