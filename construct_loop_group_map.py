"""Construct the Jarvis-Norbury loop map with a minimal direct disk solver.

    1. pull the connection back to one disk D_w,
    2. Solve D_zbar g = 0 via a collocation system with Chebyshev-Lobatto grid,
    3. Impose the condition that the boundary frame is unitary and g(z=1) = Id.
"""

import numpy as np

from convert_sadun_segert_to_wz_coords import load_sadun_segert_connection

π = np.pi

DEFAULT_W = 0.2 + 0.3j
DISK_RADIUS = 0.999
RADIAL_POINTS = 10
ANGLE_POINTS = 24

def main():
    l = 4 ; r = -5 ; t = 3
    A_mu = load_sadun_segert_connection(l, r, t, step=1e-8)
    boundary_frame = solve_D_zbar_g_eq_0(A_mu, DEFAULT_W)


# =============================================================================================


def gauge_fix_boundary_frame(boundary_frame):
    """ The remaining holomorphic gauge freedom g ↦ gη is fixed by 
        imposing the condition that g is unitary at the boundary.
    """
    pass


# =============================================================================================


def solve_D_zbar_g_eq_0(A_mu, w=DEFAULT_W, 
    disk_radius=DISK_RADIUS, radial_points=RADIAL_POINTS, angle_points=ANGLE_POINTS):
    """Construct a gauge-fixed interior frame before imposing boundary unitarity."""

    # z = rho exp(i phi)
    rho, d_rho = chebyshev_lobatto_grid(disk_radius, radial_points)
    phi = np.linspace(0.0, 2.0*π, angle_points, endpoint=False)
    d_phi = fourier_differentiation_matrix(angle_points)

    matrix_size = A_mu([w.real, w.imag, disk_radius, 0.0])[0].shape[0]
    system_matrix, boundary_data = build_collocation_system(
        A_mu, w, rho, phi, d_rho, d_phi, matrix_size
    )

    flattened_frame_grid, *_ = np.linalg.lstsq(system_matrix, boundary_data, rcond=None)
    frame_grid = flattened_frame_grid.reshape(
        radial_points, angle_points, matrix_size, matrix_size,
    )
    boundary_frame = frame_grid[-1]

    return boundary_frame


def build_collocation_system(A_mu, w, rho, phi, d_rho, d_phi, matrix_size):
    """ Turn the PDE into a linear equation of the form:
            system_matrix @ flattened_frame_grid = boundary_data 

        The point normalization below fixes the right-holomorphic gauge freedom
        to make the PDE well-posed.
    """
    n_rho = rho.size
    n_phi = phi.size
    n_unknowns = n_rho * n_phi * matrix_size
    system_matrix = np.zeros((n_unknowns, n_unknowns), dtype=complex)
    boundary_data = np.zeros((n_unknowns, matrix_size), dtype=complex)
    next_equation = 0

    # Collocation equations for D_zbar g = 0 away from rho = 0.
    for j in range(1, n_rho):
        for k, angle in enumerate(phi):
            z = rho[j] * np.exp(1j*angle)
            A_zbar = connection_zbar(A_mu, w, z)
            add_dzbar_equations(system_matrix, next_equation,
                rho, phi, d_rho, d_phi, j, k, A_zbar,
            )
            next_equation += matrix_size

    # Regularity at the disk centre: all angular samples agree at rho = 0.
    for k in range(1, n_phi):
        equation_rows = slice(next_equation, next_equation + matrix_size)
        system_matrix[equation_rows, vector_slice(0, k, n_phi, matrix_size)] = np.eye(matrix_size)
        system_matrix[equation_rows, vector_slice(0, 0, n_phi, matrix_size)] = -np.eye(matrix_size)
        next_equation += matrix_size

    # Point normalization at the disk radius: g(disk_radius, phi=0) = identity.
    equation_rows = slice(next_equation, next_equation + matrix_size)
    system_matrix[equation_rows, vector_slice(n_rho - 1, 0, n_phi, matrix_size)] = np.eye(matrix_size)
    boundary_data[equation_rows] = np.eye(matrix_size)
    next_equation += matrix_size

    assert next_equation == n_unknowns
    return system_matrix, boundary_data


def add_dzbar_equations(system_matrix, first_equation, rho, phi, d_rho, d_phi, j, k, A_zbar):
    """ Add the three equations for D_zbar g = 0 at one grid point. """
    n_phi = phi.size
    matrix_size = A_zbar.shape[0]
    equation_rows = slice(first_equation, first_equation + matrix_size)
    phase = 0.5 * np.exp(1j*phi[k])
    identity = np.eye(matrix_size)

    # 1/2 exp(i phi) d_rho g
    for radial_index, coeff in enumerate(d_rho[j]):
        system_matrix[equation_rows, vector_slice(radial_index, k, n_phi, matrix_size)] += (
            phase * coeff * identity
        )

    # 1/2 exp(i phi) (i/rho) d_phi g
    for angle_index, coeff in enumerate(d_phi[k]):
        system_matrix[equation_rows, vector_slice(j, angle_index, n_phi, matrix_size)] += (
            1j * phase * coeff * identity / rho[j]
        )

    # A_zbar g
    system_matrix[equation_rows, vector_slice(j, k, n_phi, matrix_size)] += A_zbar

# =============================================================================================


def connection_zbar(A_mu, w, z):
    coords = [w.real, w.imag, z.real, z.imag]
    A = A_mu(coords)
    return 0.5 * (np.asarray(A[2]) + 1j*np.asarray(A[3]))

def chebyshev_lobatto_grid(radius, points):
    j = np.arange(points)

    c = np.ones(points)
    c[0] = 2.0 ; c[-1] = 2.0
    c *= (-1.0)**j

    cheb_coord = np.cos(π * j / (points - 1))

    d_cheb_coord = cheb_coord[:, None] - cheb_coord[None, :]
    d_cheb = (c[:, None] / c[None, :]) / (d_cheb_coord + np.eye(points))
    d_cheb -= np.diag(np.sum(d_cheb, axis=1))

    rho = 0.5 * radius * (1.0 - cheb_coord)
    d_rho = (-2.0 / radius) * d_cheb
    return rho, d_rho


def fourier_differentiation_matrix(points):
    """ Returns matrix d_phi such that d_phi @ f approximates df/dphi """
    modes = np.fft.fftfreq(points) * points
    return np.fft.ifft(1j*modes[:, None] * np.fft.fft(np.eye(points), axis=0), axis=0)


def vector_index(j, k, a, angle_points, matrix_size):
    """ Flatten grid indices rho_j and phi_k and frame index a into one vector index. """
    return (j * angle_points + k) * matrix_size + a


def vector_slice(j, k, angle_points, matrix_size):
    start = vector_index(j, k, 0, angle_points, matrix_size)
    return slice(start, start + matrix_size)


def max_matrix_norm(matrix_list):
    return max(np.linalg.norm(matrix, ord="fro") for matrix in matrix_list)


if __name__ == "__main__":
    main()
