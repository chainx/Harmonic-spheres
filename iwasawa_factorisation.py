import numpy as np
from scipy.optimize import least_squares

from disk_solve import boundary_unitarity_residuals, basepoint_normalize_frame, timer

REFINEMENT_MAX_NFEV = 80 # Maximum number of residual evaluations used by the nonlinear least-squares inverse-factor refinement.
RETRY_MAX_UNITARITY_RESIDUAL = 1e-5

class iwasawa_factorise:
    def __init__(self, disk_frame, disk_grid, refinement_max_nfev=REFINEMENT_MAX_NFEV):
        self.verbose = False

        self.disk_frame = disk_frame
        self.boundary_frame = disk_frame[-1]
        self.disk_grid = disk_grid
        self.refinement_max_nfev = refinement_max_nfev
        self.sample_count, self.matrix_size, _ = self.boundary_frame.shape

        # compute H := γ^\dagger γ and H^{-1}
        self.metric_samples = self.boundary_frame.conj().swapaxes(-1, -2) @ self.boundary_frame
        self.inv_metric_samples = np.linalg.inv(self.metric_samples)


    def iwasawa_factorise_loop(self):
        """ Factorise loop group element γ into:
                -g which is unitary at the boundary,
                -η whose inverse is solved for holomorphically on the full disk grid.
            so γ = gη.
        """
        mode_count, η_inv_coef = self.find_optimal_mode_count()
        η_inv_coef = self.refine_η_inv(η_inv_coef)
        
        η_inv = self.extend_η_inv_to_disk(η_inv_coef)

        gauge_fixed_disk_frame = self.disk_frame @ η_inv
        gauge_fixed_disk_frame, η_inv = basepoint_normalize_frame(gauge_fixed_disk_frame, η_inv)

        return gauge_fixed_disk_frame, η_inv, boundary_unitarity_residuals(gauge_fixed_disk_frame[-1])

    def find_optimal_mode_count(self):
        max_mode_count = self.sample_count//2 - 1
        best_mode_count, best_η_inv_coef, best_residual = None, None, 1e10
        for mode_count in range(1, max_mode_count):
            boundary_η_inv, η_inv_coef = self.toeplitz_cholesky_factorisation(mode_count)
            new_boundary_frame = self.boundary_frame @ boundary_η_inv
            residual = boundary_unitarity_residuals(new_boundary_frame)["avg_unitarity_residual"]
            if residual < best_residual:
                best_mode_count = mode_count; best_η_inv_coef = η_inv_coef; best_residual = residual
        if self.verbose: print(f"Max mode count: {max_mode_count}. Best mode count: {best_mode_count}")
        return best_mode_count, best_η_inv_coef

    def toeplitz_cholesky_factorisation(self, mode_count):
        toeplitz_matrix = self.construct_truncated_toeplitz_matrix(mode_count)
        cholesky_matrix = np.linalg.cholesky(toeplitz_matrix) # T(H^{-1}) = L L^\dagger
        η_inv_coef = self.extract_η_inv_coef(cholesky_matrix)
        boundary_η_inv = self.evaluate_η_inv_on_boundary(η_inv_coef)
        return boundary_η_inv, η_inv_coef

    def construct_truncated_toeplitz_matrix(self, mode_count):
        """Constructs (mode_count+1)x(mode_count+1) Toepliz matrix from the sampled loop group map values."""
        fourier_coef = self.extract_inv_metric_fourier_coefficients(mode_count)

        block_count = mode_count + 1
        toeplitz_matrix = np.zeros((block_count * self.matrix_size, block_count * self.matrix_size), dtype=complex)
        for i in range(block_count):
            row = slice(i * self.matrix_size, (i + 1) * self.matrix_size)
            for j in range(block_count):
                col = slice(j * self.matrix_size, (j + 1) * self.matrix_size)
                toeplitz_matrix[row, col] = fourier_coef[i - j + mode_count]

        return toeplitz_matrix

    def extract_inv_metric_fourier_coefficients(self, mode_count):
        """Returns first mode_count +ve and -ve Taylor coefficients (which are matrices) of H^{-1}."""
        fourier_coef = np.fft.fft(self.inv_metric_samples, axis=0) / self.sample_count
        return np.concatenate((fourier_coef[-mode_count:], fourier_coef[:mode_count+1]))

    def extract_η_inv_coef(self, cholesky_matrix):
        """Extract coefficients of η^{-1} from a Toeplitz Cholesky factor."""
        block_count = cholesky_matrix.shape[0] // self.matrix_size
        last_block_row = slice((block_count - 1) * self.matrix_size, block_count * self.matrix_size)
        
        η_inv_coeff = np.empty((block_count, self.matrix_size, self.matrix_size), dtype=complex)
        for mode in range(block_count):
            col = slice((block_count - 1 - mode) * self.matrix_size, (block_count - mode) * self.matrix_size)
            η_inv_coeff[mode] = cholesky_matrix[last_block_row, col]

        return η_inv_coeff

    def evaluate_η_inv_on_boundary(self, η_inv_coef):
        """ Evaluate η^{-1} on the boundary circle. """
        modes = np.arange(η_inv_coef.shape[0])
        phi = np.linspace(0.0, 2.0*np.pi, self.sample_count, endpoint=False)
        phases = np.exp(1j * phi[:, None] * modes[None, :])
        return np.einsum("kn,nab->kab", phases, η_inv_coef)

    def extend_η_inv_to_disk(self, η_inv_coef):
        """ Evaluate η^{-1} on the polar disk grid. """
        modes = np.arange(η_inv_coef.shape[0]) # [0,1,2,3, ... ,mode_count]
        phi = np.linspace(0.0, 2.0*np.pi, self.sample_count, endpoint=False)
        normalized_radii = self.disk_grid / self.disk_grid[-1]
        z = normalized_radii[:, None] * np.exp(1j * phi[None, :])
        z_powers = z[:, :, None] ** modes[None, None, :]
        return np.einsum("jkn,nab->jkab", z_powers, η_inv_coef)

    def refine_η_inv(self, η_inv_coef):
        """ Minimize the boundary unitarity residual over holomorphic Fourier coefficients. """
        η_inv_coef = np.ascontiguousarray(η_inv_coef, dtype=complex)

        def boundary_unitarity_residual(real_coefficients):
            candidate_η_inv_coef = real_coefficients.view(complex).reshape(η_inv_coef.shape)
            candidate_η_inv = self.evaluate_η_inv_on_boundary(candidate_η_inv_coef)
            return boundary_unitarity_residuals(self.boundary_frame @ candidate_η_inv, flatten_for_optimizer=True)

        initial_coefficients = η_inv_coef.view(float).ravel()

        result = least_squares(boundary_unitarity_residual, initial_coefficients,
            method="trf", max_nfev=self.refinement_max_nfev, x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
        )
        return result.x.view(complex).reshape(η_inv_coef.shape).copy()
