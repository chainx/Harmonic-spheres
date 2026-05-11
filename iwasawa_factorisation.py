import numpy as np
from scipy.optimize import least_squares

MODE_COUNT = 127 # The number of positive (and negative) Fourier modes to keep for the Toeplitz matrix truncation
REFINEMENT_MAX_NFEV = 80 # Maximum number of residual evaluations used by the nonlinear least-squares inverse-factor refinement.

def main():
    saved_frame = np.load("results/jarvis_norbury_large_frame.npz")
    loop_samples = saved_frame["boundary_frame"]
    center_value = saved_frame["frame_grid"][0, 0]
    g, eta = iwasawa_decompose_loop(loop_samples, saved_frame["rho"], center_value=center_value, refinement_max_nfev=REFINEMENT_MAX_NFEV)
    verify_g_unitary_at_boundary(g, avg_tol=1e-1, max_tol=1e-1)


# =============================================================================================


def iwasawa_decompose_loop(loop_samples, disk_radii, refinement_max_nfev=REFINEMENT_MAX_NFEV, center_value=None, mode_count=None):
    """ Decompose loop group element into:
            -g which is unitary at the boundary,
            -eta whose inverse is solved for holomorphically on the full disk grid.
    """
    sample_count, matrix_size, _ = loop_samples.shape
    if not mode_count: mode_count = sample_count//2 - 1
    
    metric_samples = np.einsum("kab,kac->kbc", loop_samples.conj(), loop_samples) # compute H = g^\dagger g
    toeplitz_matrix = construct_truncated_toeplitz_matrix(metric_samples, mode_count, sample_count, matrix_size)
    cholesky_matrix = np.linalg.cholesky(toeplitz_matrix)
    boundary_eta, _ = extract_holomorphic_factor(cholesky_matrix, sample_count, matrix_size)
    boundary_q, q_coeff = refine_inverse_holomorphic_factor(
        invert_matrix_samples(boundary_eta), metric_samples, mode_count, refinement_max_nfev,
    )
    g = np.einsum("kab,kbc->kac", loop_samples, boundary_q)
    return g, invert_matrix_samples(evaluate_holomorphic_factor_on_disk(q_coeff, disk_radii, sample_count))

def construct_truncated_toeplitz_matrix(loop_samples, mode_count, sample_count, matrix_size):
    """ Constructs (mode_count+1)x(mode_count+1) Toepliz matrix from the sampled loop group map values. """
    fourier_coef = extract_fourier_coefficients(loop_samples, mode_count, sample_count)
    block_count = mode_count + 1
    toeplitz_matrix = np.zeros((block_count * matrix_size, block_count * matrix_size), dtype=complex)
    for i in range(block_count):
        row = slice(i * matrix_size, (i + 1) * matrix_size)
        for j in range(block_count):
            col = slice(j * matrix_size, (j + 1) * matrix_size)
            toeplitz_matrix[row, col] = fourier_coef[i - j + mode_count]

    return toeplitz_matrix

def extract_fourier_coefficients(loop_samples, mode_count, sample_count):
    """ Returns first mode_count +ve and -ve Taylor coefficients (which are matrices) of the sampled analytic loop. """
    if mode_count >= sample_count // 2:
        raise ValueError(
            f"mode_count must be less than {sample_count // 2} for {sample_count} samples"
        )
    fourier_coef = np.fft.fft(loop_samples, axis=0) / sample_count
    return np.concatenate((fourier_coef[-mode_count:], fourier_coef[:mode_count+1]))

def extract_holomorphic_factor(cholesky_matrix, sample_count, matrix_size):
    """ Extract boundary value of holomorphic matrix-valued function eta from a Toeplitz Cholesky factor. """
    block_count = cholesky_matrix.shape[0] // matrix_size
    last_block_row = slice((block_count - 1) * matrix_size, block_count * matrix_size)
    eta_coeff = np.empty((block_count, matrix_size, matrix_size), dtype=complex)

    for mode in range(block_count):
        col = slice((block_count - 1 - mode) * matrix_size, (block_count - mode) * matrix_size)
        eta_coeff[mode] = cholesky_matrix[last_block_row, col]

    modes = np.arange(eta_coeff.shape[0])
    phi = np.linspace(0.0, 2.0*np.pi, sample_count, endpoint=False)
    phases = np.exp(1j * phi[:, None] * modes[None, :])
    return np.einsum("kn,nab->kab", phases, eta_coeff), eta_coeff

def refine_inverse_holomorphic_factor(q, metric_samples, mode_count, max_nfev):
    """ Minimize the boundary unitarity residual over holomorphic Fourier coefficients. """
    sample_count, matrix_size, _ = q.shape
    coefficient_shape = (mode_count + 1, matrix_size, matrix_size)
    modes = np.arange(mode_count + 1)
    phi = np.linspace(0.0, 2.0*np.pi, sample_count, endpoint=False)
    phases = np.exp(1j * phi[:, None] * modes[None, :])

    initial_coefficients = np.fft.fft(q, axis=0)[:mode_count + 1] / sample_count
    coefficient_size = np.prod(coefficient_shape)

    def pack_coefficients(coefficients):
        return np.concatenate([coefficients.real.ravel(), coefficients.imag.ravel()])

    def unpack_coefficients(packed_coefficients):
        real_part = packed_coefficients[:coefficient_size].reshape(coefficient_shape)
        imaginary_part = packed_coefficients[coefficient_size:].reshape(coefficient_shape)
        return real_part + 1j*imaginary_part

    def evaluate_holomorphic_factor(coefficients):
        return np.einsum("kn,nab->kab", phases, coefficients)

    def boundary_unitarity_residual(packed_coefficients):
        candidate_q = evaluate_holomorphic_factor(unpack_coefficients(packed_coefficients))
        candidate_metric = np.einsum(
            "kba,kbc,kcd->kad",
            candidate_q.conj(), metric_samples, candidate_q,
        )
        residual = candidate_metric - np.eye(matrix_size)
        return np.concatenate([residual.real.ravel(), residual.imag.ravel()])

    result = least_squares(boundary_unitarity_residual, pack_coefficients(initial_coefficients),
        method="trf", max_nfev=max_nfev, x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    q_coeff = unpack_coefficients(result.x)
    return evaluate_holomorphic_factor(q_coeff), q_coeff

def evaluate_holomorphic_factor_on_disk(eta_coeff, disk_radii, sample_count):
    """ Evaluate eta on the polar disk grid. """
    modes = np.arange(eta_coeff.shape[0])
    phi = np.linspace(0.0, 2.0*np.pi, sample_count, endpoint=False)
    normalized_radii = disk_radii / disk_radii[-1]
    powers = normalized_radii[:, None, None]**modes[None, None, :] * np.exp(1j * phi[None, :, None] * modes[None, None, :])
    return np.einsum("jkn,nab->jkab", powers, eta_coeff)

def invert_matrix_samples(matrix_samples):
    """ Invert a matrix-valued grid samplewise. """
    return np.linalg.inv(matrix_samples)

def multiply_by_pointwise_inverse_on_right(left_factors, right_factors):
    """ Return left_factors @ inv(right_factors) without explicitly forming inverses. """
    product = np.empty_like(left_factors)
    for sample_index, (left_factor, right_factor) in enumerate(zip(left_factors, right_factors)):
        product[sample_index] = np.linalg.solve(right_factor.T, left_factor.T).T
    return product

def verify_g_unitary_at_boundary(g, avg_tol, max_tol):
    """ Check the average and max Frobenius norms of the residuals g g^\dagger - I. """
    matrix_size = g.shape[-1]
    metric = np.einsum("kab,kac->kbc", g.conj(), g)
    residuals = np.linalg.norm(metric - np.eye(matrix_size), axis=(1, 2))
    avg_residual = np.mean(residuals)
    max_residual = np.max(residuals)

    print(f"average boundary unitarity residual = {avg_residual:.6e}")
    print(f"max boundary unitarity residual = {max_residual:.6e}")
    return avg_residual < avg_tol and max_residual < max_tol


# =============================================================================================


if __name__ == "__main__":
    main()
