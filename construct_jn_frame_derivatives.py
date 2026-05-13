"""Compute sphere derivatives of the Jarvis-Norbury frame.

For alpha = u,v, set Θ_alpha = g^{-1} d_alpha g.  The derivative problem is

    d_zbar Θ_alpha = -g^{-1}(d_alpha A_zbar)g,
    Θ_alpha + Θ_alpha^* = 0 on |z|=R,
    Θ_alpha(R, 0) = 0.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from construct_bpst_in_wz_coords import load_bpst_connection
from construct_loop_group_map import (
    DEFAULT_W,
    DISK_RADIUS,
    RADIAL_POINTS,
    ANGLE_POINTS,
    STEP,
    chebyshev_lobatto_grid,
    connection_zbar,
    multiply_by_pointwise_inverse_on_right,
    solve_D_zbar_g_eq_0,
)
from iwasawa_factorisation import iwasawa_decompose_loop


SPHERE_SHIFTS = {"u": 1.0, "v": 1j}

def main():
    A_mu = load_bpst_connection()
    g, dg, Θ = compute_jn_frame_derivatives(A_mu)
    print("frame shape:", g.shape)
    print("max |d_w g|_F:", np.linalg.norm(dg["w"], axis=(-2, -1)).max())
    print("u/v basepoint Θ residual:", max(np.linalg.norm(Θ[a][-1, 0]) for a in ("u", "v")))



def compute_jn_frame_derivatives(
    A_mu, w=DEFAULT_W, disk_radius=DISK_RADIUS,
    radial_points=RADIAL_POINTS, angle_points=ANGLE_POINTS, step=STEP,
):
    """Return the normalized frame g, derivatives dg, and pulled back Maurer-Cartan form Θ."""
    rho, phi, g = normalized_jn_frame(A_mu, w, disk_radius, radial_points, angle_points)
    Θ = {alpha: solve_Θ(A_mu, w, g, rho, phi, step, alpha) for alpha in ("u", "v")}
    dg = {alpha: g @ Θ_alpha for alpha, Θ_alpha in Θ.items()}
    dg["w"] = 0.5 * (dg["u"] - 1j * dg["v"])
    return g, dg, Θ

def normalized_jn_frame(A_mu, w, disk_radius, radial_points, angle_points):
    """Construct the JN frame and right-normalize it so g(w, R)=Id at phi=0."""
    rho = chebyshev_lobatto_grid(disk_radius, radial_points)[0]
    phi = np.linspace(0.0, 2.0 * np.pi, angle_points, endpoint=False)
    raw = solve_D_zbar_g_eq_0(A_mu, w, disk_radius, radial_points, angle_points)
    _, eta = iwasawa_decompose_loop(raw[-1], rho, center_value=raw[0, 0])
    g = multiply_by_pointwise_inverse_on_right(raw, eta)
    return rho, phi, g @ np.linalg.inv(g[-1, 0])

def solve_Θ(A_mu, w, g, rho, phi, step, alpha):
    """Solve for Θ_alpha = g^{-1} d_alpha g."""
    source = -np.linalg.solve(g, dA_zbar(A_mu, w, rho, phi, step, alpha) @ g)
    return solve_dbar(source, rho, phi)

def dA_zbar(A_mu, w, rho, phi, step, alpha):
    """Central finite difference for d_u A_zbar or d_v A_zbar."""
    if hasattr(A_mu, "dA_zbar"):
        out = np.empty((rho.size, phi.size, *connection_zbar(A_mu, w, rho[-1]).shape), complex)
        for j, r in enumerate(rho):
            for k, angle in enumerate(phi):
                out[j, k] = A_mu.dA_zbar(w, r * np.exp(1j * angle), step, alpha)
        return out

    shift = SPHERE_SHIFTS[alpha] * step
    out = np.empty((rho.size, phi.size, *connection_zbar(A_mu, w, rho[-1]).shape), complex)
    for j, r in enumerate(rho):
        for k, angle in enumerate(phi):
            z = r * np.exp(1j * angle)
            out[j, k] = (
                connection_zbar(A_mu, w + shift, z)
                - connection_zbar(A_mu, w - shift, z)
            ) / (2.0 * step)
    return out

def solve_dbar(source, rho, phi, mode_cutoff=None):
    """Invert d_zbar by Fourier modes and impose skew/basepoint boundary data."""
    Θ_part, mode_cutoff = particular_dbar_solution(source, rho, phi, mode_cutoff)
    return Θ_part + skew_boundary_correction(Θ_part, rho, phi, mode_cutoff)

def particular_dbar_solution(source, rho, phi, mode_cutoff=None):
    """Regular particular solution of d_zbar Θ=source."""
    if mode_cutoff is None:
        mode_cutoff = rho.size // 2
    mode_cutoff = min(mode_cutoff, max(0, phi.size // 2 - 2))
    source_hat = np.fft.fft(source, axis=1) / phi.size

    Θ_part = np.zeros_like(source)
    for n in range(-mode_cutoff, mode_cutoff + 1):
        Θ_n = regular_radial_solution(source_hat[:, (n + 1) % phi.size], rho, n)
        Θ_part += Θ_n[:, None] * np.exp(1j * n * phi)[None, :, None, None]
    return Θ_part, mode_cutoff

def regular_radial_solution(f, rho, mode):
    """Solve Θ_n' - n Θ_n/rho = 2f regularly at rho=0."""
    weight = np.zeros_like(rho, dtype=float)
    if mode == 0:
        weight[0] = 1.0
    weight[1:] = rho[1:] ** (-mode)
    integral = cumulative_trapezoid(f * weight[:, None, None], rho, axis=0, initial=0.0)

    Θ = np.zeros((rho.size, *f.shape[1:]), dtype=complex)
    Θ[1:] = 2.0 * rho[1:, None, None] ** mode * integral[1:]
    return Θ

def skew_boundary_correction(Θ_part, rho, phi, mode_cutoff):
    """Holomorphic correction enforcing Θ+Θ^\dagger=0 and Θ(R,0)=0."""
    boundary_fix = -(Θ_part[-1] + adjoint(Θ_part[-1]))
    h = holomorphic_extension(boundary_fix, rho, phi, mode_cutoff)
    h -= 0.5 * (np.fft.fft(boundary_fix, axis=0)[0] / phi.size)[None, None]
    return h - (Θ_part[-1, 0] + h[-1, 0])[None, None]

def holomorphic_extension(boundary_value, rho, phi, mode_cutoff):
    """Project boundary data to nonnegative Fourier modes and extend inward."""
    coeff = np.fft.fft(boundary_value, axis=0) / phi.size
    extension = np.empty((rho.size, phi.size, *boundary_value.shape[1:]), dtype=complex)
    extension[:] = coeff[0]
    for n in range(1, mode_cutoff + 1):
        z_to_n = (rho[:, None] / rho[-1]) ** n * np.exp(1j * n * phi[None, :])
        extension += z_to_n[..., None, None] * coeff[n]
    return extension

def adjoint(a):
    return a.conj().swapaxes(-1, -2)


if __name__ == "__main__":
    main()
