"""Construct the Jarvis-Norbury loop map with a minimal direct disk solver.

    1. pull the connection back to one disk D_w,
    2. Solve D_zbar g = 0 via a collocation system with Chebyshev-Lobatto grid,
    3. Impose the condition that the boundary frame is unitary and g(z=1) = Id.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from construct_bpst_in_wz_coords import load_bpst_connection
from construct_sadun_segert_in_wz_coords import load_sadun_segert_connection

from disk_solve import disk_solver, timer
from iwasawa_factorisation import iwasawa_factorise

def main():
    A_mu = load_bpst_connection()
    # A_mu = load_sadun_segert_connection(l=4, r=-5, t=3)
    disk_frame, η_inv = JN_frame_solver(A_mu).construct_JN_frame(w=0.2 + 0.3j)

class JN_frame_solver(disk_solver):
    def construct_JN_frame(self, w):
        system_matrix, boundary_data = self.build_collocation_system(w)
        disk_frame, boundary_frame = self.solve_PDE(w, system_matrix, boundary_data)
        print("PDE residual before holomorphic gauge fixing")
        print(self.PDE_residual(disk_frame, system_matrix, boundary_data))

        gauge_fixed_disk_frame, η_inv, unitarity_residuals = iwasawa_factorise(disk_frame, self.radial_grid).iwasawa_factorise_loop()
        print("JN frame constructed!\n", unitarity_residuals)
        print(self.PDE_residual(gauge_fixed_disk_frame, system_matrix, boundary_data))
        return gauge_fixed_disk_frame, η_inv

    @timer
    def solve_PDE(self, w, system_matrix, boundary_data):
        flattened_disk_frame = spsolve(system_matrix.tocsc(), boundary_data)
        disk_frame = flattened_disk_frame.reshape(
            self.radial_points, self.angular_points, self.matrix_size, self.matrix_size,
        )
        boundary_frame = disk_frame[-1]
        return disk_frame, boundary_frame
    
    @timer
    def build_collocation_system(self, w):
        """ Turn the PDE into a linear equation of the form:
                system_matrix @ flattened_disk_frame = boundary_data 

            The point normalization below fixes the right-holomorphic gauge freedom
            to make the PDE well-posed.
        """
        n_unknowns = self.radial_points * self.angular_points * self.matrix_size
        system_matrix = lil_matrix((n_unknowns, n_unknowns), dtype=complex)
        boundary_data = np.zeros((n_unknowns, self.matrix_size), dtype=complex)
        next_equation = 0

        # Collocation equations for D_zbar g = 0 away from r = 0.
        for j in range(1, self.radial_points):
            for k, angle in enumerate(self.angular_grid):
                z = self.radial_grid[j] * np.exp(1j * angle)
                self.add_PDE_equations(w, z, system_matrix, next_equation, j, k)
                next_equation += self.matrix_size

        # Regularity at the disk centre: all angular samples agree at r = 0.
        for k in range(1, self.angular_points):
            equation_rows = slice(next_equation, next_equation + self.matrix_size)
            system_matrix[equation_rows, self.vector_slice(0, k)] = np.eye(self.matrix_size)
            system_matrix[equation_rows, self.vector_slice(0, 0)] = -np.eye(self.matrix_size)
            next_equation += self.matrix_size

        # Point normalization at the disk boundary: g(r=disk_radius, φ=0) = identity.
        equation_rows = slice(next_equation, next_equation + self.matrix_size)
        system_matrix[equation_rows, self.vector_slice(self.radial_points - 1, 0)] = np.eye(self.matrix_size)
        boundary_data[equation_rows] = np.eye(self.matrix_size)
        next_equation += self.matrix_size

        assert next_equation == n_unknowns
        return system_matrix, boundary_data
    
    def add_PDE_equations(self, w, z, system_matrix, next_equation, j, k):
        equation_rows = slice(next_equation, next_equation + self.matrix_size)

        phase = 0.5 * np.exp(1j*self.angular_grid[k])
        identity = np.eye(self.matrix_size)

        # 1/2 exp(i φ) dr g
        radial_cols = (
            self.matrix_size * (np.arange(self.radial_points) * self.angular_points + k)[:, None]
            + np.arange(self.matrix_size)
        ).ravel()
        system_matrix[equation_rows, radial_cols] += phase * np.kron(self.radial_deriv[j], identity)

        # 1/2 exp(i φ) (i/r) dφ g
        angular_cols = slice(self.vector_slice(j, 0).start, self.vector_slice(j, self.angular_points - 1).stop)
        system_matrix[equation_rows, angular_cols] += \
            1j * phase * np.kron(self.angular_deriv[k], identity) / self.radial_grid[j]

        # A_zbar g
        system_matrix[equation_rows, self.vector_slice(j, k)] += \
            self.connection_zbar(w, z)

    def PDE_residual(self, disk_frame, system_matrix, boundary_data):
        flattened_disk_frame = disk_frame.reshape(
            self.radial_points * self.angular_points * self.matrix_size, self.matrix_size,
        )
        residual = system_matrix @ flattened_disk_frame - boundary_data
        return {
            "max_PDE_residual": np.max(np.abs(residual)),
            "frobenius_PDE_residual": np.linalg.norm(residual),
        }

if __name__ == "__main__":
    main()
