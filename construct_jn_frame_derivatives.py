"""Compute sphere derivatives of the Jarvis-Norbury frame.

For alpha = u,v, set Θ_alpha = g^{-1} d_alpha g.  The derivative problem is

    d_zbar Θ_alpha = -g^{-1}(d_alpha A_zbar)g,
    Θ_alpha + Θ_alpha^* = 0 on |z|=R,
    Θ_alpha(R, 0) = 0.
"""

import numpy as np

from construct_bpst_in_wz_coords import load_bpst_connection

from disk_solve import ANGULAR_POINTS, DISK_RADIUS, RADIAL_POINTS
from jarvis_norbury_frame import JN_frame_solver


DEFAULT_W = 0.2 + 0.3j
STEP = 5e-4
SPHERE_SHIFTS = {"u": 1.0, "v": 1j}

def main():
    A_mu = load_bpst_connection()
    g, dg, Θ = JN_derivatives(A_mu).compute_jn_frame_derivatives(w=DEFAULT_W)
    print("max |d_w g|_F:", np.linalg.norm(dg["w"], axis=(-2, -1)).max())
    print("u/v basepoint Θ residual:", max(np.linalg.norm(Θ[a][-1, 0]) for a in ("u", "v")))

class JN_derivatives(JN_frame_solver):
    def __init__(
        self,
        A_mu,
        radial_points=RADIAL_POINTS,
        angular_points=ANGULAR_POINTS,
        disk_radius=DISK_RADIUS,
        step=STEP,
    ):
        super().__init__(A_mu, radial_points, angular_points, disk_radius)
        self.step = step

    def compute_jn_frame_derivatives(self, w=DEFAULT_W):
        """Return the normalized frame g, derivatives dg, and pulled back Maurer-Cartan form Θ."""
        g, _ = self.construct_JN_frame(w)
        Θ = {alpha: self.solve_Θ(w, g, alpha) for alpha in ("u", "v")}
        dg = {alpha: g @ Θ_alpha for alpha, Θ_alpha in Θ.items()}
        dg["w"] = 0.5 * (dg["u"] - 1j * dg["v"])
        return g, dg, Θ

    def solve_Θ(self, w, g, alpha):
        """Solve for Θ_alpha = g^{-1} d_alpha g."""
        source = -np.linalg.solve(g, self.dA_zbar(w, alpha) @ g)
        return self.solve_dzbar_equals_source(source)

    def solve_second_Θ(self, w, g, Θ_alpha, alpha):
        """Solve for partial_alpha Θ_alpha by differentiating the dbar equation."""
        d_alpha_A_zbar = self.dA_zbar(w, alpha)
        conjugated_d_alpha_A_zbar = np.linalg.solve(g, d_alpha_A_zbar @ g)
        conjugated_d2_alpha_A_zbar = np.linalg.solve(
            g, self.d2A_zbar(w, alpha) @ g
        )
        source = (
            Θ_alpha @ conjugated_d_alpha_A_zbar
            - conjugated_d_alpha_A_zbar @ Θ_alpha
            - conjugated_d2_alpha_A_zbar
        )
        return self.solve_dzbar_equals_source(source)

    def solve_dzbar_equals_source(self, source, mode_cutoff=None):
        """Invert d_zbar by Fourier modes and impose skew/basepoint boundary data."""
        Θ_part, mode_cutoff = self.particular_dbar_solution(source, mode_cutoff)
        return Θ_part + self.skew_boundary_correction(Θ_part, mode_cutoff)

    def particular_dbar_solution(self, source, mode_cutoff=None):
        """Regular particular solution of d_zbar Θ=source."""

        if mode_cutoff is None:
            mode_cutoff = self.radial_points - 2
        mode_cutoff = min(mode_cutoff, max(0, self.angular_points // 2 - 2))
        source_hat = np.fft.fft(source, axis=1) / self.angular_points

        Θ_part = np.zeros_like(source)
        for mode in range(-mode_cutoff, mode_cutoff + 1):
            f = source_hat[:, (mode + 1) % self.angular_points]
            Θ_n = self.radial_collocation_solution(f, mode)
            phase = np.exp(1j * mode * self.angular_grid)
            Θ_part += Θ_n[:, None] * phase[None, :, None, None]
        return Θ_part, mode_cutoff

    def radial_collocation_solution(self, f, mode):
        """Solve Θ_n' - n Θ_n/r = 2f(r) with a fixed holomorphic ambiguity."""

        operator = self.radial_deriv.astype(complex).copy()
        for j in range(1, self.radial_points):
            operator[j, j] -= mode / self.radial_grid[j]

        rhs = 2.0 * f.copy()
        rhs[0] = 0

        operator[0] = 0
        if mode > 0:
            operator[0, -1] = 1
        else:
            operator[0, 0] = 1

        flat_rhs = rhs.reshape(self.radial_points, -1)
        return np.linalg.solve(operator, flat_rhs).reshape(f.shape)

    def skew_boundary_correction(self, Θ_part, mode_cutoff):
        """Holomorphic correction enforcing Θ+Θ^\dagger=0 and Θ(R,0)=0."""
        boundary_fix = -(Θ_part[-1] + adjoint(Θ_part[-1]))
        h = self.holomorphic_extension(boundary_fix, mode_cutoff)
        h -= 0.5 * (np.fft.fft(boundary_fix, axis=0)[0] / self.angular_points)[None, None]
        return h - (Θ_part[-1, 0] + h[-1, 0])[None, None]

    def holomorphic_extension(self, boundary_value, mode_cutoff):
        """Project boundary data to nonnegative Fourier modes and extend inward."""
        coeff = np.fft.fft(boundary_value, axis=0) / self.angular_points
        extension = np.empty(
            (self.radial_points, self.angular_points, *boundary_value.shape[1:]),
            dtype=complex,
        )
        extension[:] = coeff[0]
        for n in range(1, mode_cutoff + 1):
            z_to_n = (
                (self.radial_grid[:, None] / self.radial_grid[-1]) ** n
                * np.exp(1j * n * self.angular_grid[None, :])
            )
            extension += z_to_n[..., None, None] * coeff[n]
        return extension

    def dA_zbar(self, w, alpha, step=None):
        """Central finite difference for d_u A_zbar or d_v A_zbar."""
        if step is None:
            step = self.step

        out = np.empty(
            (self.radial_points, self.angular_points, self.matrix_size, self.matrix_size),
            dtype=complex,
        )

        if hasattr(self.A_mu, "dA_zbar"):
            for j, r in enumerate(self.radial_grid):
                for k, angle in enumerate(self.angular_grid):
                    z = r * np.exp(1j * angle)
                    out[j, k] = self.A_mu.dA_zbar(w, z, step, alpha)
            return out

        shift = SPHERE_SHIFTS[alpha] * step
        for j, r in enumerate(self.radial_grid):
            for k, angle in enumerate(self.angular_grid):
                z = r * np.exp(1j * angle)
                out[j, k] = (
                    self.connection_zbar(w + shift, z)
                    - self.connection_zbar(w - shift, z)
                ) / (2.0 * step)
        return out

    def d2A_zbar(self, w, alpha):
        """Second central difference of A_zbar in u or v."""

        out = np.empty(
            (self.radial_points, self.angular_points, self.matrix_size, self.matrix_size),
            dtype=complex,
        )

        if hasattr(self.A_mu, "d2A_zbar"):
            for j, r in enumerate(self.radial_grid):
                for k, angle in enumerate(self.angular_grid):
                    z = r * np.exp(1j * angle)
                    out[j, k] = self.A_mu.d2A_zbar(w, z, self.step, alpha)
            return out

        shift = SPHERE_SHIFTS[alpha] * self.step
        for j, r in enumerate(self.radial_grid):
            for k, angle in enumerate(self.angular_grid):
                z = r * np.exp(1j * angle)
                out[j, k] = (
                    self.connection_zbar(w + shift, z)
                    - 2.0 * self.connection_zbar(w, z)
                    + self.connection_zbar(w - shift, z)
                ) / self.step**2
        return out

def adjoint(a):
    return a.conj().swapaxes(-1, -2)


if __name__ == "__main__":
    main()
