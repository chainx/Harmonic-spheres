"""Convert a saved reduced solution from ym_reduced.py in the (w,z) coordinates, 
where w=u+iv on S^2 and z=x+iy on D^2.  
   
This file reconstructs

    A = A_u du + A_v dv + A_x dx + A_y dy

from the saved Sadun-Segert function a_3.
"""

from pathlib import Path
RESULTS_PATH = Path.cwd() / "results"

import numpy as np

from construct_sadun_segert_ym_solution import construct_a3_and_da3_from_fourier_coeff

π = np.pi

# Lie algebra generators of so(3)
T1 = np.matrix([[0, 0, 0], [0, 0,-1], [ 0, 1, 0]])
T2 = np.matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
T3 = np.matrix([[0,-1, 0], [1, 0, 0], [ 0, 0, 0]])
T = [T1, T2, T3]
# which satisy [T1, T2] = T3, [T2, T3] = T1, and [T3, T1] = T2
assert np.array_equal(T1 * T2 - T2 * T1, T3)
assert np.array_equal(T2 * T3 - T3 * T2, T1)
assert np.array_equal(T3 * T1 - T1 * T3, T2)

# Basis for symmetric traceless 3x3 matrices
Q0 = np.matrix([[-1, 0, 0], [0,-1, 0], [ 0, 0, 2]])
Q1 = np.matrix([[ 0, 0, 1], [0, 0, 0], [-1, 0, 0]])
Q2 = np.matrix([[ 0, 0, 0], [0, 0, 0], [ 0, 0, 0]])
Q3 = np.matrix([[ 0, 0, 0], [0, 0, 0], [ 0, 0, 0]])
Q4 = np.matrix([[ 0, 0, 0], [0, 0, 0], [ 0, 0, 0]])
Q = [Q0, Q1, Q2, Q3, Q4]

def main():
    l = 1 # Truncates to keep the first 2l + 6 positive and negative Fourier modes
    r = -5 ; t = 3 # Inputs defining boundary conditions which uniquely define a_3(0)
    fourier_coeff = np.load(RESULTS_PATH/f"r={r}_t={t}_solution_with_{2*(2*l+6)}_modes.npy")
    a3, _ = construct_a3_and_da3_from_fourier_coeff(fourier_coeff) # Reconstruct a_3(0)

    A_mu = connection_in_wz_coords(a3)


# =============================================================================================


def paper_wz_to_s4(u, v, x, y):
    """ Map paper coordinates (w,z)=(u+iv,x+iy) to S^4 as a subset of R^5 """

    w = complex(u, v)
    z = complex(x, y)
    z2 = float((z.conjugate() * z).real)
    if z2 > 1.0:
        raise ValueError("need |z| ≤ 1")
    w2 = float((w.conjugate() * w).real)
    denominator = 1.0 + w2 * z2
    a = w * (1.0 - z2) / denominator
    b = z * (1.0 + w2) / denominator
    q2 = float((a.conjugate() * a + b.conjugate() * b).real)
    return np.array(
        [
            (q2 - 1.0) / (1.0 + q2),
            2.0 * a.real / (1.0 + q2),
            2.0 * a.imag / (1.0 + q2),
            2.0 * b.real / (1.0 + q2),
            2.0 * b.imag / (1.0 + q2),
        ]
    )

def s4_to_matrix(v):
    return sum(vi * Qi for vi, Qi in zip(v, Q))

def extract_θ_and_g(M):
    # M is real symmetric and traceless, since it is in the span of Q
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Put eigenvalues in descending order to get unique diagonal matrix
    order = np.argsort(eigenvalues)[::-1]
    diag_M = eigenvalues[order]
    
    g = eigenvectors[:, order]
    # Force g into SO(3) by ensuring it has positive determinant
    if np.linalg.det(g) < 0:
        g[:, -1] *= -1

    # Recover theta from the diagonal matrix
    sin_θ = 0.5 * (diag_M[1] - diag_M[2]); cos_θ = 0.5 * np.sqrt(3.0) * diag_M[0]
    sin_θ = np.clip(sin_θ, -1.0, 1.0); cos_θ = np.clip(cos_θ, -1.0, 1.0) # Numerical safety
    θ = np.arctan2(sin_θ, cos_θ)
    θ = np.clip(θ, 0.0, π/3) # Numerical safety

    return θ, g

def gauge_fix(g, g_base, theta_base, eps=1e-2):
    # Near theta = 0, first column is the stable distinguished axis
    if theta_base < eps:
        g_fixed = degenerate_gauge_fix(g, g_base, 0)
    # Near theta = pi/3, third column is the stable distinguished axis
    if np.pi/3 - theta_base < eps:
        g_fixed = degenerate_gauge_fix(g, g_base, 2)

    # Fix the discrete sign ambiguity away from the singular set
    reflections = [
        np.diag([ 1,  1,  1]),
        np.diag([ 1, -1, -1]),
        np.diag([-1,  1, -1]),
        np.diag([-1, -1,  1]),
    ]
    g_fixed = min(
        [g_fixed @ R for R in reflections],
        key=lambda h: np.linalg.norm(h - g_base, ord='fro')
    )
    return g_fixed

def normalize(v): return v / np.linalg.norm(v)
def project_perp(v, u): return v - np.dot(v, u) * u
def match_sign(v, v_ref): return v if np.dot(v, v_ref) >= 0 else -v

def degenerate_gauge_fix(g, g_base, primary_axis):
    e1 = match_sign(g[:, primary_axis], g_base[:, primary_axis])
    e1 = normalize(e1)

    # Continue the old second vector into the new plane e1^⊥
    e2 = project_perp(g_base[:, 1], e1)
    e2 = normalize(e2)

    e3 = np.cross(e1, e2)
    e3 = normalize(e3)

    return np.column_stack([e1, e2, e3])

def maurer_cartan_form(coords, step=1e-10):
    """ Return (g^{-1}dg)_mu """

    M = s4_to_matrix(paper_wz_to_s4(*coords))
    θ, g = extract_θ_and_g(M)

    g_inv_dg = []
    for mu in range(4):
        coords_new = coords.copy()
        coords_new[mu] += step
        M_new = s4_to_matrix(paper_wz_to_s4(*coords_new))
        _, g_new = extract_θ_and_g(M_new)
        g_new = gauge_fix(g_new, g, θ)
        g_inv_dg.append( ( g.T @ (g_new - g) ) / step )

    return g_inv_dg, θ

def extract_basis_vector_coeff(lie_alg_matrix):
    return [np.trace(lie_alg_matrix @ Q[i])/2 for i in range(3)]

def connection_in_wz_coords(a3, step=1e-10):
    def A_mu(coords):
        g_inv_dg, θ = maurer_cartan_form(coords, step)
        a = [a3(θ - 2*π/3), a3(θ + 2*π/3), a3(θ)]
        A = []
        for mu in range(4):
            coeff = extract_basis_vector_coeff(g_inv_dg[mu])
            A[mu] = sum([coeff[i] * a[i] * Q[i] for i in range(3)])
        return A
    return A_mu


# =============================================================================================


if __name__ == "__main__":
    main()
