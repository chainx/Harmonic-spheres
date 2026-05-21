from functools import wraps
from time import perf_counter
import numpy as np
π = np.pi

DISK_RADIUS = 0.999

RADIAL_POINTS = 16
ANGULAR_POINTS = 100

class disk_solver:
    def __init__(self, A_mu, radial_points=RADIAL_POINTS, angular_points=ANGULAR_POINTS, disk_radius=DISK_RADIUS):
        self.disk_radius = disk_radius
        self.radial_points = radial_points
        self.angular_points = angular_points

        self.radial_grid, self.radial_deriv = self.chebyshev_lobatto_grid()
        self.angular_grid = np.linspace(0, 2*π, self.angular_points, endpoint=False)
        self.angular_deriv = self.fourier_differentiation_matrix()

        self.A_mu = A_mu
        self.matrix_size = getattr(A_mu, "matrix_size", None)
        if self.matrix_size is None:
            self.matrix_size = A_mu([0.2, 0.3, 0.1, 0.0])[0].shape[0]

    def chebyshev_lobatto_grid(self):
        """Returns Chebyshev-Lobatto grid for the interval [0, width],
           and matrix d_r such that d_r @ f approximates df/dr."""
        j = np.arange(self.radial_points)

        grid = np.cos(π * j / (self.radial_points - 1)) # Runs from 1 to -1, fixed later

        c = np.ones(self.radial_points) # c = [2, -1, 1, ..., (+/-)2]
        c[0] = 2.0 ; c[-1] = 2.0 ; c *= (-1.0)**j
        C = (c[:, None] / c[None, :]) # C_ij = c_i / c_j

        grid_diffs = grid[:, None] - grid[None, :] # (grid_diffs)_ij = x_i - x_j
        # (d_Cheb)_ij = (c_i / c_j) / (x_i-x_j) when i≠j else 0
        d_cheb = C / (grid_diffs + np.eye(self.radial_points)) 
        d_cheb -= np.diag(np.sum(d_cheb, axis=1))
        deriv_matrix = (-2.0 / self.disk_radius) * d_cheb 

        grid = 0.5 * self.disk_radius * (1.0 - grid) # Rescaling and fixing orientation

        return grid, deriv_matrix

    def fourier_differentiation_matrix(self):
        """Returns matrix dφ such that dφ @ f approximates df/dφ."""
        modes = np.fft.fftfreq(self.angular_points) * self.angular_points # [0,1,...,N_points//2-1, -N_points//2,...]
        mode_matrix = np.fft.fft(np.eye(self.angular_points), axis=0) # (mode_matrix)_kl = exp(-2πi kl/N_points)
        mode_matrix_deriv = 1j*modes[:, None] * mode_matrix # exp(ikφ) → ik exp(ikφ)
        return np.fft.ifft(mode_matrix_deriv, axis=0) # Inverse fourier transform back to position space

    def flatten_indices(self, j, k, a):
        """ Flatten grid indices r_j and φ_k and matrix index a into one vector index. """
        return (j * self.angular_points + k) * self.matrix_size + a

    def vector_slice(self, j, k):
        start = self.flatten_indices(j, k, 0)
        return slice(start, start + self.matrix_size)
    


    def connection_zbar(self, w, z):
        if hasattr(self.A_mu, "zbar_at"):
            return self.A_mu.zbar_at(w, z)
        coords = [w.real, w.imag, z.real, z.imag]
        A = self.A_mu(coords)
        return 0.5 * (np.asarray(A[2]) + 1j*np.asarray(A[3]))



def basepoint_normalize_frame(disk_frame, eta_inv=None):
    """Normalize the final frame so g(z=1)=Id, preserving g = γ η^{-1}."""
    basepoint_value = disk_frame[-1, 0]
    basepoint_inverse = np.linalg.inv(basepoint_value)
    normalized_frame = disk_frame @ basepoint_inverse
    if eta_inv is None:
        return normalized_frame
    return normalized_frame, eta_inv @ basepoint_inverse

def boundary_unitarity_residuals(g, flatten_for_optimizer=False):
    matrix_size = g.shape[-1]
    metric = g.conj().swapaxes(-1, -2) @ g
    residual = metric - np.eye(matrix_size)
    if flatten_for_optimizer:
        return np.ascontiguousarray(residual).view(float).ravel()

    residuals = np.linalg.norm(residual, axis=(1, 2))
    return {"max_unitarity_residual": np.max(residuals), "avg_unitarity_residual": np.mean(residuals)}

def max_matrix_norm(matrix_list):
    return max(np.linalg.norm(matrix, ord="fro") for matrix in matrix_list)

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_time = perf_counter() - start_time
            print(f"{func.__qualname__} ran in {elapsed_time:.3f}s")
    return wrapper
