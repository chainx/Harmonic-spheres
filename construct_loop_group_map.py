"""Construct the Jarvis-Norbury loop map with a minimal direct disk solver.

    1. pull the connection back to one disk D_w,
    2. Solve D_zbar g = 0 via a collocation system with Chebyshev-Lobatto grid,
    3. Impose the condition that the boundary frame is unitary and g(z=1) = Id.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from datetime import datetime
from pathlib import Path
RESULTS_PATH = Path.cwd() / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

from construct_sadun_segert_in_wz_coords import load_sadun_segert_connection
from construct_bpst_in_wz_coords import load_bpst_connection
from iwasawa_factorisation import iwasawa_decompose_loop, verify_g_unitary_at_boundary

π = np.pi

DEFAULT_W = 0.2 + 0.3j
STEP = 5e-4

DISK_RADIUS = 0.999

RADIAL_POINTS = 36
ANGLE_POINTS = 500

def main():
    # A_mu = load_sadun_segert_connection(l=4, r=-5, t=3, step=1e-8)
    A_mu = load_bpst_connection()

    radial_resolution = [20,30,40,50,60]
    angular_resolution = [200,300,400,500,600,700,800]

    print(f"Starting at {datetime.now().strftime('%H:%M:%S')}")
    for radial_points in radial_resolution:
        for angle_points in angular_resolution:
            rho = chebyshev_lobatto_grid(DISK_RADIUS, radial_points)[0]
            phi = np.linspace(0.0, 2.0*np.pi, angle_points, endpoint=False)
            frame_grid = solve_D_zbar_g_eq_0(A_mu, DEFAULT_W, radial_points=radial_points, angle_points=angle_points)
            loop_samples = frame_grid[-1]
            gauge_fixed_loop_samples, eta = iwasawa_decompose_loop(loop_samples, rho, center_value=frame_grid[0, 0])
            gauge_fixed_frame_grid = multiply_by_pointwise_inverse_on_right(frame_grid, eta)
            gauge_fixed_frame_grid, eta = basepoint_normalize_frame(gauge_fixed_frame_grid, eta)
            gauge_fixed_loop_samples = gauge_fixed_frame_grid[-1]
            np.savez(RESULTS_PATH/f"bpst_JN_frame_w={DEFAULT_W}_{radial_points}-{angle_points}.npz", 
                     g=gauge_fixed_frame_grid, eta=eta, rho=rho, phi=phi, w=DEFAULT_W
            )
            print(f"Finished radial_points={radial_points}, angle_points={angle_points} at {datetime.now().strftime('%H:%M:%S')}")
        
            verify_g_unitary_at_boundary(gauge_fixed_loop_samples, avg_tol=1e-4, max_tol=1e-4)

# =============================================================================================


def solve_D_zbar_g_eq_0(A_mu, w=DEFAULT_W, 
    disk_radius=DISK_RADIUS, radial_points=RADIAL_POINTS, angle_points=ANGLE_POINTS):
    """Return the full disk frame before imposing boundary unitarity.

    The returned array has shape
    (radial_points, angle_points, matrix_size, matrix_size).
    """

    # z = rho exp(i phi)
    rho, d_rho = chebyshev_lobatto_grid(disk_radius, radial_points)
    phi = np.linspace(0.0, 2.0*π, angle_points, endpoint=False)
    d_phi = fourier_differentiation_matrix(angle_points)

    matrix_size = A_mu([w.real, w.imag, disk_radius, 0.0])[0].shape[0]
    system_matrix, boundary_data = build_collocation_system(
        A_mu, w, rho, phi, d_rho, d_phi, matrix_size
    )

    flattened_disk_frame = spsolve(system_matrix.tocsc(), boundary_data)
    frame_grid = flattened_disk_frame.reshape(
        radial_points, angle_points, matrix_size, matrix_size,
    )
    return frame_grid


# =============================================================================================


def build_collocation_system(A_mu, w, rho, phi, d_rho, d_phi, matrix_size):
    """ Turn the PDE into a linear equation of the form:
            system_matrix @ flattened_frame_grid = boundary_data 

        The point normalization below fixes the right-holomorphic gauge freedom
        to make the PDE well-posed.
    """
    n_rho = rho.size
    n_phi = phi.size
    n_unknowns = n_rho * n_phi * matrix_size
    system_matrix = lil_matrix((n_unknowns, n_unknowns), dtype=complex)
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

def multiply_by_pointwise_inverse_on_right(left_factors, right_factors):
    """ Return left_factors @ inv(right_factors) over a matrix-valued grid. """
    return np.linalg.solve(right_factors.swapaxes(-1, -2), left_factors.swapaxes(-1, -2)).swapaxes(-1, -2)


def basepoint_normalize_frame(frame_grid, eta=None):
    """Normalize the final frame so g(z=1)=Id, preserving frame = raw @ inv(eta)."""
    basepoint_value = frame_grid[-1, 0]
    normalized_frame = frame_grid @ np.linalg.inv(basepoint_value)
    if eta is None:
        return normalized_frame
    return normalized_frame, basepoint_value @ eta


if __name__ == "__main__":
    main()
