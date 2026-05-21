"""Check the based-loop harmonic equation in a truncated Fourier model.

For the Grassmannian model W = g H_+, the projector equation is

    [P, P_uu + P_vv] = 0

where P = M_γ P_+ M_γ^{-1} with γ the boundary value of the JN frame g.

Conjugating by g reduces this to a fixed P_+ computation. If
Θ_a = g^{-1} \partial_a g and Θ_aa = partial_a Θ_a, then the relevant
second-derivative term is

    S = [Θ_uu + Θ_vv, P_+]
        + [Θ_u, [Θ_u, P_+]]
        + [Θ_v, [Θ_v, P_+]].

The quadratic commutators are the Christoffel/second-fundamental-form terms,
and the harmonic equation becomes

    [P_+, S] = 0
"""

import numpy as np

from construct_bpst_in_wz_coords import load_bpst_connection
from construct_sadun_segert_in_wz_coords import load_sadun_segert_connection

from construct_jn_frame_derivatives import (
    DEFAULT_W, DISK_RADIUS, STEP, JN_derivatives,
)

def main():
    # A_mu = load_bpst_connection()
    A_mu = load_sadun_segert_connection(l=4, r=-5, t=3)
    residual = projector_harmonic_residual(A_mu)
    print(residual)



def projector_harmonic_residual(A_mu,
    w=DEFAULT_W, radial_points=16, angle_points=100,
    disk_radius=DISK_RADIUS, step=STEP, operator_modes=None,
    iwasawa_mode_count=None,
):
    """Return || [P_+, S] || for the truncated loop multiplication operators."""

    derivative_solver = JN_derivatives(
        A_mu,
        radial_points=radial_points,
        angular_points=angle_points,
        disk_radius=disk_radius,
        step=step,
    )
    g, _, Θ = derivative_solver.compute_jn_frame_derivatives(w)
    Θ_uu = derivative_solver.solve_second_Θ(w, g, Θ["u"], "u")
    Θ_vv = derivative_solver.solve_second_Θ(w, g, Θ["v"], "v")

    Θ_u, Θ_v, dΘ_trace = Θ["u"][-1], Θ["v"][-1], Θ_uu[-1] + Θ_vv[-1]

    sample_count, matrix_size, _ = Θ_u.shape
    if operator_modes is None:
        operator_modes = min(8, sample_count // 4 - 1)

    P_plus = hardy_projection(operator_modes, matrix_size)
    Θ_u = multiplication_operator(Θ_u, operator_modes)
    Θ_v = multiplication_operator(Θ_v, operator_modes)
    dΘ_trace = multiplication_operator(dΘ_trace, operator_modes)

    S = (
        commutator(dΘ_trace, P_plus)
        + commutator(Θ_u, commutator(Θ_u, P_plus))
        + commutator(Θ_v, commutator(Θ_v, P_plus))
    )
    residual = commutator(P_plus, S)
    absolute = np.linalg.norm(residual, ord="fro")
    scale = max(np.linalg.norm(S, ord="fro"), 1.0)
    return {
        "operator_modes": operator_modes,
        "absolute": float(absolute),
        "relative": float(absolute / scale),
    }

def multiplication_operator(samples, mode_radius):
    """Matrix for truncated multiplication by a matrix-valued loop."""
    sample_count, matrix_size, _ = samples.shape
    modes = np.arange(-mode_radius, mode_radius + 1)
    coeff = np.fft.fft(samples, axis=0) / sample_count
    operator = np.zeros((modes.size * matrix_size, modes.size * matrix_size), complex)

    for row, out_mode in enumerate(modes):
        row_slice = block(row, matrix_size)
        for col, in_mode in enumerate(modes):
            operator[row_slice, block(col, matrix_size)] = coeff[(out_mode - in_mode) % sample_count]
    return operator

def hardy_projection(mode_radius, matrix_size):
    """Projection onto nonnegative Fourier modes in the truncated basis."""
    modes = np.arange(-mode_radius, mode_radius + 1)
    P = np.zeros((modes.size * matrix_size, modes.size * matrix_size), complex)
    for index, mode in enumerate(modes):
        if mode >= 0:
            P[block(index, matrix_size), block(index, matrix_size)] = np.eye(matrix_size)
    return P

def block(index, matrix_size):
    start = index * matrix_size
    return slice(start, start + matrix_size)

def commutator(a, b):
    return a @ b - b @ a


if __name__ == "__main__":
    main()
