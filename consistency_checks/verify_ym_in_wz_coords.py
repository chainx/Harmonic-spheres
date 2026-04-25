import numpy as np

from convert_sadun_segert_to_wz_coords import load_sadun_segert_connection, random_sadun_segert_connection

DIFF_STEP = 5e-6 # Default step size for numerical differentiation

YM_TOL = 1e-5 # Maximum discrepancy allowed before an error is thrown
N_POINTS = 200 # Number of points to check the Yang-Mills equation at

def main():
    A_mu = load_sadun_segert_connection(l=4, r=-5, t=3, step=DIFF_STEP)
    # A_mu = random_sadun_segert_connection() # Verify the check isn't trivially solved

    # for coord in sample_points(N_POINTS):
    #     ratio = compute_ym_residual_ratio(A_mu, coord, verbose=True, step=DIFF_STEP)
    #     assert ratio < YM_TOL
    # print("connection satisfies the Yang-Mills equation at the sampled points")

    # For (l=4, r=-5, t=3) this point shows the importance of gauge fixing g when computing g^{-1}dg
    coord = [
        -0.0055024614286730495,
        -1.0336795493686264,
        0.029999999754185668,
        -3.840424450273706e-06
    ]
    ratio = compute_ym_residual_ratio(A_mu, coord, verbose=True, step=DIFF_STEP)
    assert ratio < 1e-6, ratio

# =============================================================================================


def array_connection(A_mu, coords):
    return [np.asarray(matrix) for matrix in A_mu(coords)]

def curvature(A_mu, coords, step=1e-6):
    coords = np.array(coords, dtype=float)
    A = array_connection(A_mu, coords)

    dA = []
    for mu in range(4):
        coords_plus = coords.copy() ; coords_plus[mu] += step
        coords_minus = coords.copy() ; coords_minus[mu] -= step
        A_plus = array_connection(A_mu, coords_plus)
        A_minus = array_connection(A_mu, coords_minus)
        dA.append([(A_plus[nu] - A_minus[nu]) / (2*step) for nu in range(4)])

    matrix_shape = A[0].shape
    matrix_dtype = np.result_type(*[matrix.dtype for matrix in A])
    F = [[np.zeros(matrix_shape, dtype=matrix_dtype) for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(mu + 1, 4):
            F[mu][nu] = dA[mu][nu] - dA[nu][mu] + A[mu] @ A[nu] - A[nu] @ A[mu]
            F[nu][mu] = -F[mu][nu]
    return F

def metric_diagonal(coords):
    u, v, x, y = coords
    sphere_factor = 4 / (1 + u*u + v*v)**2
    disk_factor = 4 / (1 - x*x - y*y)**2
    return np.array([sphere_factor, sphere_factor, disk_factor, disk_factor])

def sqrt_det_metric(coords):
    sphere_factor, _, disk_factor, _ = metric_diagonal(coords)
    return sphere_factor * disk_factor

def yang_mills_residual(A_mu, coords, step=1e-6):
    coords = np.array(coords, dtype=float)
    A = array_connection(A_mu, coords)
    F = curvature(A_mu, coords, step)
    g_inv = 1 / metric_diagonal(coords)
    sqrt_g = sqrt_det_metric(coords)

    residual = []
    for nu in range(4):
        total = np.zeros(A[0].shape, dtype=np.result_type(*[matrix.dtype for matrix in A]))
        for mu in range(4):
            if mu == nu:
                continue

            coords_plus = coords.copy() ; coords_plus[mu] += step
            coords_minus = coords.copy() ; coords_minus[mu] -= step

            g_inv_plus = 1 / metric_diagonal(coords_plus)
            g_inv_minus = 1 / metric_diagonal(coords_minus)
            F_plus = curvature(A_mu, coords_plus, step)
            F_minus = curvature(A_mu, coords_minus, step)

            density_plus = sqrt_det_metric(coords_plus) * g_inv_plus[mu] * g_inv_plus[nu] * F_plus[mu][nu]
            density_minus = sqrt_det_metric(coords_minus) * g_inv_minus[mu] * g_inv_minus[nu] * F_minus[mu][nu]

            total += (density_plus - density_minus) / (2*step*sqrt_g)

            F_raised = g_inv[mu] * g_inv[nu] * F[mu][nu]
            total += A[mu] @ F_raised - F_raised @ A[mu]

        residual.append(total)
    return residual

def compute_ym_residual_ratio(A_mu, coord, verbose=True, step=1e-6):
    F = curvature(A_mu, coord, step)
    residual = yang_mills_residual(A_mu, coord, step)

    curvature_norm = max_matrix_norm([F[mu][nu] for mu in range(4) for nu in range(mu + 1, 4)])
    residual_norm = max_matrix_norm(residual)
    ratio = residual_norm / curvature_norm

    if verbose:
        print(f"{coord = }")
        print(f"{curvature_norm = }")
        print(f"{residual_norm = }")
        print(f"{ratio = }\n\n")
    
    return ratio
    

# =============================================================================================


def max_matrix_norm(matrix_list):
    return max(np.linalg.norm(matrix, ord='fro') for matrix in matrix_list)

def sample_points(N_points, seed=0):
    rng = np.random.default_rng(seed)

    w_centres = np.array([
        [ 0.0 ,  0.0 ],
        [ 0.2 ,  0.3 ],
        [ 0.35, -0.1 ],
        [-0.25,  0.1 ],
        [ 0.1 , -0.35],
        [ 0.8 ,  0.2 ],
        [-0.6 , -0.5 ],
    ])
    disk_radii = np.array([0.05, 0.2, 0.45, 0.7, 0.9, 0.98])

    points = []
    while len(points) < N_points:
        w = w_centres[len(points) % len(w_centres)] + rng.uniform(-0.04, 0.04, size=2)
        radius = disk_radii[len(points) % len(disk_radii)]
        angle = rng.uniform(0, 2*np.pi)
        z = radius * np.array([np.cos(angle), np.sin(angle)])
        coord = np.concatenate([w, z])
        points.append(coord.tolist())
    return points

if __name__ == "__main__":
    main()
