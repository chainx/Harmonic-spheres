import numpy as np

MODE_COUNT = 120 # The number of positive (and negative) Fourier modes to keep for the Toeplitz matrix truncation

def main():
    loop_samples = np.load("results/jarvis_norbury_large_frame.npz")["boundary_frame"]
    g, eta = iwasawa_decompose_loop(loop_samples, mode_count=MODE_COUNT)
    verify_g_unitary_at_boundary(g, avg_tol=1e-1, max_tol=1e-1)


# =============================================================================================


def iwasawa_decompose_loop(loop_samples, mode_count):
    """ Decompose loop group element into:
            -g which is unitary at the boundary,
            -eta which is the boundary value of a holomorphic function from the interior.
    """
    sample_count, matrix_size, _ = loop_samples.shape
    metric_samples = np.einsum("kab,kac->kbc", loop_samples.conj(), loop_samples)
    toeplitz_matrix = construct_truncated_toeplitz_matrix(metric_samples, mode_count, sample_count, matrix_size)
    cholesky_matrix = np.linalg.cholesky(toeplitz_matrix)
    eta = extract_eta(cholesky_matrix, sample_count, matrix_size)
    g = np.einsum("kab,kbc->kac", loop_samples, np.linalg.inv(eta))
    return g, eta

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

def extract_eta(cholesky_matrix, sample_count, matrix_size):
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
    return np.einsum("kn,nab->kab", phases, eta_coeff)

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
