import numpy as np

from construct_sadun_segert_in_wz_coords import S

CYCLES = {(0, 1, 2), (1, 2, 0), (2, 0, 1)}

def eta_symbol(a, mu, nu):
    epsilon_part = mu < 3 and nu < 3 and (((a, mu, nu) in CYCLES) - ((a, nu, mu) in CYCLES))
    return epsilon_part + (nu == 3 and mu == a) - (mu == 3 and nu == a)

ETA = np.array([[[eta_symbol(a, mu, nu) for nu in range(4)] for mu in range(4)] for a in range(3)])

def stereographic_coords_and_jacobian(coords):
    u, v, x, y = coords
    w = np.array([u, v]); z = np.array([x, y])
    w_norm_sq = w @ w; z_norm_sq = z @ z
    sphere_factor = 1 - z_norm_sq; disk_factor = 1 + w_norm_sq
    denominator = 1 + w_norm_sq*z_norm_sq

    stereographic_coords = np.r_[w*sphere_factor/denominator, z*disk_factor/denominator]
    dw_block = sphere_factor*np.eye(2)/denominator - 2*sphere_factor*z_norm_sq*np.outer(w, w)/denominator**2
    dz_block = disk_factor*np.eye(2)/denominator - 2*disk_factor*w_norm_sq*np.outer(z, z)/denominator**2
    mixed_wz_block = -2*disk_factor*np.outer(w, z)/denominator**2
    mixed_zw_block = 2*sphere_factor*np.outer(z, w)/denominator**2
    return stereographic_coords, np.block([[dw_block, mixed_wz_block], [mixed_zw_block, dz_block]])

def load_bpst_connection(rho=1):
    def A_mu(coords):
        X, jacobian = stereographic_coords_and_jacobian(coords)
        denominator = X @ X + rho*rho
        bpst_in_r4 = [sum(2*ETA[a, mu, nu]*X[nu]*S[a]/denominator for a in range(3) for nu in range(4)) for mu in range(4)]
        return [sum(jacobian[mu, alpha]*bpst_in_r4[mu] for mu in range(4)) for alpha in range(4)]
    return A_mu
