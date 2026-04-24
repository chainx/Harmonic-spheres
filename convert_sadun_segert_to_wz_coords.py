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

DIFF_STEP = 5e-8 # Default step size for numerical differentiation

# Lie algebra generators of so(3)
T1 = np.matrix([[0, 0, 0], [0, 0,-1], [ 0, 1, 0]])
T2 = np.matrix([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
T3 = np.matrix([[0,-1, 0], [1, 0, 0], [ 0, 0, 0]])
T = [T1, T2, T3]
# which satisy [T1, T2] = T3, [T2, T3] = T1, and [T3, T1] = T2
assert np.array_equal(T1 * T2 - T2 * T1, T3)
assert np.array_equal(T2 * T3 - T3 * T2, T1)
assert np.array_equal(T3 * T1 - T1 * T3, T2)

# Basis for the 5-dimensional space of real symmetric traceless 3x3 matrices.
# These are mutually orthogonal and all have the same Frobenius norm.
Q0 = np.matrix([[-1, 0, 0], [0,-1, 0], [0, 0, 2]]) * (1/np.sqrt(3))
Q1 = np.matrix([[ 0, 0, 1], [0, 0, 0], [1, 0, 0]])
Q2 = np.matrix([[ 0, 0, 0], [0, 0, 1], [0, 1, 0]])
Q3 = np.matrix([[ 1, 0, 0], [0,-1, 0], [0, 0, 0]])
Q4 = np.matrix([[ 0, 1, 0], [1, 0, 0], [0, 0, 0]])
Q = [Q0, Q1, Q2, Q3, Q4]

def main():
    l = 4 # Truncates to keep the first 2l + 6 positive and negative Fourier modes
    r = -5 ; t = 3 # Inputs defining boundary conditions which uniquely define a_3(0)
    A_mu = load_sadun_segert_connection(l, r, t, step=DIFF_STEP)

    coords = [
        [0, 0, 0, 0],         # exactly at z = 0
        [1, 0, 0, 0],         # another point on the z = 0 orbit
        [1, 1, 0, 1],         # exactly at |z| = 1
        [1, 1, 0, 0.999],     # just inside the |z| = 1 orbit
        [0.2, 0.3, 0.001, 0], # close to z = 0, but generic
        [0.2, 0.3, 0.1, 0.4], # generic interior point
    ]
    for coord in coords:
        print(f"{coord = }")
        for lie_alg_matrix in A_mu(coord):
            print(lie_alg_matrix)


# =============================================================================================


def wz_to_s4(u, v, x, y):
    """ Map paper coordinates (w,z)=(u+iv,x+iy) to S^4 as a subset of R^5 """

    w = complex(u, v)
    z = complex(x, y)
    z2 = np.clip(z.conjugate() * z, 0.0, 1.0)
    w2 = w.conjugate() * w
    denominator = 1.0 + w2 * z2
    a = w * (1.0 - z2) / denominator
    b = z * (1.0 + w2) / denominator
    q2 = float((a.conjugate() * a + b.conjugate() * b).real)
    return np.array(
        [
            (q2 - 1.0)   / (1.0 + q2),
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

    # Q_θ = cos(θ) Q0 + sin(θ) Q3 is diagonal with entries
    # (-cos(θ)/sqrt(3) + sin(θ), -cos(θ)/sqrt(3) - sin(θ), 2cos(θ)/sqrt(3))
    # np.linalg.eigh returns these in ascending eigenvalue order, 
    # which for for θ in [0, π/3] yields [second, first, third] 
    order = [1, 0, 2]
    diag_M = eigenvalues[order]
    
    g = np.asarray(eigenvectors[:, order], dtype=float)
    # Force g into SO(3) by ensuring it has positive determinant
    if np.linalg.det(g) < 0:
        g[:, -1] *= -1

    # Recover theta from the diagonal matrix
    sin_θ = 0.5 * (diag_M[0] - diag_M[1]); cos_θ = 0.5 * np.sqrt(3.0) * diag_M[2]
    sin_θ = np.clip(sin_θ, -1.0, 1.0); cos_θ = np.clip(cos_θ, -1.0, 1.0) # Numerical safety
    θ = np.arctan2(sin_θ, cos_θ)
    θ = np.clip(θ, 0.0, π/3) # Numerical safety

    return θ, g

def gauge_fix(g, g_base, theta_base, eps=1e-2):
    # Near theta = 0, first column is the stable distinguished axis
    if theta_base < eps:
        g = degenerate_gauge_fix(g, g_base, 0)
    # Near theta = pi/3, third column is the stable distinguished axis
    if np.pi/3 - theta_base < eps:
        g = degenerate_gauge_fix(g, g_base, 2)

    # Fix the discrete sign ambiguity away from the singular set
    reflections = [
        np.diag([ 1,  1,  1]),
        np.diag([ 1, -1, -1]),
        np.diag([-1,  1, -1]),
        np.diag([-1, -1,  1]),
    ]
    g = min(
        [g @ R for R in reflections],
        key=lambda h: np.linalg.norm(h - g_base, ord='fro')
    )
    return g

def normalize(v): return v / np.linalg.norm(v)
def project_perp(v, u): return v - np.dot(v, u) * u

def degenerate_gauge_fix(g, g_base, primary_axis):
    e2_base = np.array(g_base[:, 1]).reshape(-1)
    e1 = np.array(g[:, primary_axis]).reshape(-1)

    e2 = project_perp(e2_base, e1)
    e3 = np.cross(e1, e2)
    
    e1 = normalize(e1); e2 = normalize(e2); e3 = normalize(e3)

    if primary_axis == 0:
        return np.column_stack([e1, e2, e3])
    elif primary_axis == 2:
        return np.column_stack([e3, e2, e1])
    else:
        raise ValueError("primary_axis must be either 0 or 2")

def maurer_cartan_form(coords, step=DIFF_STEP):
    """ Return (g^{-1}dg)_mu """

    M = s4_to_matrix(wz_to_s4(*coords))
    θ, g = extract_θ_and_g(M)
    # Put the base frame in a deterministic reflection representative too.
    # Otherwise separate calls to A_mu(coords) can choose different base gauges.
    g = gauge_fix(g, np.eye(3), θ)

    g_inv_dg = []
    for mu in range(4):
        coords_plus = coords.copy()
        coords_minus = coords.copy()
        coords_plus[mu] += step
        coords_minus[mu] -= step

        M_plus = s4_to_matrix(wz_to_s4(*coords_plus))
        M_minus = s4_to_matrix(wz_to_s4(*coords_minus))
        _, g_plus = extract_θ_and_g(M_plus)
        _, g_minus = extract_θ_and_g(M_minus)

        g_plus = gauge_fix(g_plus, g, θ)
        g_minus = gauge_fix(g_minus, g, θ)
        g_inv_dg.append( ( g.T @ (g_plus - g_minus) ) / (2*step) )

    return g_inv_dg, θ

def extract_basis_vector_coeff(lie_alg_matrix):
    return [-np.trace(lie_alg_matrix @ T[i])/2 for i in range(3)]

def connection_in_wz_coords(a3, step=DIFF_STEP):
    def A_mu(coords):
        g_inv_dg, θ = maurer_cartan_form(coords, step)
        a = [a3(θ - 2*π/3), a3(θ + 2*π/3), a3(θ)]
        A = []
        for mu in range(4):
            coeff = extract_basis_vector_coeff(g_inv_dg[mu])
            A.append( -sum([coeff[i] * a[i] * T[i] for i in range(3)]) )
        return A
    return A_mu


# =============================================================================================


def load_sadun_segert_connection(l, r, t, step=DIFF_STEP):
    fourier_coeff = np.load(RESULTS_PATH/f"r={r}_t={t}_solution_with_{2*(2*l+6)}_modes.npy")
    a3, _ = construct_a3_and_da3_from_fourier_coeff(fourier_coeff)
    return connection_in_wz_coords(a3, step)

def random_sadun_segert_connection(step=DIFF_STEP, n_modes=16, seed=0):
    rng = np.random.default_rng(seed)
    a3, _ = construct_a3_and_da3_from_fourier_coeff(rng.normal(size=n_modes))
    return connection_in_wz_coords(a3, step)

if __name__ == "__main__":
    main()
