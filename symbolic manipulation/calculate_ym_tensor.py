import sympy as sp
from sympy_helper_module import simp, numerator, impose_relations
from functools import cache
from time import perf_counter

matrix_size = 2 # Gauge group U(matrix_size)

w, w_bar, z, z_bar = sp.symbols("w w_bar z z_bar")
coordinates = {"w": w, "w_bar": w_bar, "z": z, "z_bar": z_bar}
conjugate_coordinate = {"w": "w_bar", "w_bar": "w", "z": "z_bar", "z_bar": "z"}
coordinate_order = ("w", "z", "w_bar", "z_bar") # Choice of orientation convention

zero = sp.zeros(matrix_size)
s4_conformal_factor = ((1 - z*z_bar) / (1 + z*z_bar))**2

def main():
    H = matrix_functions("H")
    eta = matrix_functions("eta")

    # A general connection and corresponding degen connection in the Jarvis-Norbury gauge
    A = original_connection(H, eta)
    A_degen = degenerate_connection(eta)

    # Constraints from ASD and the boundary condition H = Id when |z|^2 = 1
    self_duality_relations = get_self_duality_relations(A, H, A_degen)
    print("Self-duality relations computed")

    # Verify the corresponding degenerate connection satisfies the degenerate YM equations
    for index in coordinates:
        residual = degen_ym_tensor(A_degen, index, self_duality_relations)
        assert is_zero_matrix(residual), residual
    print("Degenerate YM equations verified!")

# ===============   Helper functions   ================

def is_zero_matrix(matrix):
    return all(sp.simplify(entry) == 0 for entry in matrix)

def matrix_functions(name):
    return sp.Matrix(matrix_size, matrix_size, lambda row, col:
        sp.Function(f"{name}_{row + 1}{col + 1}")(w, w_bar, z, z_bar))

def normalize_conjugates(expr):
    return expr.xreplace({
        sp.conjugate(w): w_bar,
        sp.conjugate(w_bar): w,
        sp.conjugate(z): z_bar,
        sp.conjugate(z_bar): z,
    })

def normalize_conjugates_matrix(matrix):
    return matrix.applyfunc(normalize_conjugates)

# ===============   Constructing original and degenerate connection   ================

def original_connection(H, eta):
    eta_dagger = normalize_conjugates_matrix(eta.H)
    return {
        "z_bar": zero,
        "z": H.inv() * derivative(H, "z"),
        "w_bar": eta,
        "w": H.inv() * derivative(H, "w") - H.inv() * eta_dagger * H,
    }

def degenerate_connection(eta):
    eta_dagger = normalize_conjugates_matrix(eta.H)
    return {
        "z_bar": zero,
        "z": zero,
        "w_bar": eta,
        "w": -eta_dagger,
    }

# ===============   Metrics and Levi-Civita connection   ================

def metric_factor(metric):
    return s4_conformal_factor if metric == "s4" else 1

@cache
def g(index1, index2, metric="product"):
    if (index1, index2) in [("w", "w_bar"), ("w_bar", "w")]:
        return 2 * metric_factor(metric) / (1 + w*w_bar)**2
    if (index1, index2) in [("z", "z_bar"), ("z_bar", "z")]:
        return 2 * metric_factor(metric) / (1 - z*z_bar)**2
    return 0

@cache
def inv_g(index1, index2, metric="product"):
    if (index1, index2) in [("w", "w_bar"), ("w_bar", "w")]:
        return sp.Rational(1, 2) * (1 + w*w_bar)**2 / metric_factor(metric)
    if (index1, index2) in [("z", "z_bar"), ("z_bar", "z")]:
        return sp.Rational(1, 2) * (1 - z*z_bar)**2 / metric_factor(metric)
    return 0

@cache
def christoffel(index1, index2, index3, metric="product"):
    return sp.simplify(sp.Rational(1, 2) * sum(
        inv_g(index1, dummy_index, metric) * (
            sp.diff(g(index3, dummy_index, metric), coordinates[index2])
            + sp.diff(g(index2, dummy_index, metric), coordinates[index3])
            - sp.diff(g(index2, index3, metric), coordinates[dummy_index])
        )
        for dummy_index in coordinates
    ))

# ===============   Self-duality   ================

def epsilon(index1, index2, index3, index4):
    return sp.LeviCivita(*[coordinate_order.index(index) for index in [index1, index2, index3, index4]])

def hodge_star_curvature(connection, index1, index2, raised_curvature=None, metric="product"):
    if (index1, index2) in [("w", "z"), ("w_bar", "z_bar")]:
        return compute_curvature(connection, index1, index2)
    if (index2, index1) in [("w", "z"), ("w_bar", "z_bar")]:
        return -compute_curvature(connection, index2, index1)
    if raised_curvature is None:
        raised_curvature = compute_raised_curvature(connection, metric)
    volume_density = g("w", "w_bar", metric) * g("z", "z_bar", metric)
    star_F = sp.zeros(matrix_size)
    for index3 in coordinates:
        for index4 in coordinates:
            star_F += sp.Rational(1, 2) * volume_density * epsilon(index1, index2, index3, index4) * raised_curvature(index3, index4)
    return star_F

def self_duality_tensor(connection, index1, index2, raised_curvature=None, metric="product"):
    F = compute_curvature(connection, index1, index2)
    star_F = hodge_star_curvature(connection, index1, index2, raised_curvature, metric)
    if (index1, index2) in [("w", "z"), ("w_bar", "z_bar")]:
        return (F + star_F).applyfunc(simp)
    return (F + star_F).applyfunc(numerator)

def get_self_duality_relations(connection, H, reduced_connection):
    """ Returns relations from ASD of the original connection and the boundary framing """
    raised_curvature = compute_raised_curvature(connection)
    reduced_sphere_curvature = compute_curvature(reduced_connection, "w", "w_bar")
    relations = []
    wz_self_duality          = self_duality_tensor(connection, "w", "z", raised_curvature)
    w_bar_z_bar_self_duality = self_duality_tensor(connection, "w_bar", "z_bar", raised_curvature)
    sphere_self_duality      = self_duality_tensor(connection, "w", "w_bar", raised_curvature)
    relations += list(wz_self_duality)
    relations += list((H * wz_self_duality * H.inv()).applyfunc(simp))
    relations += list(w_bar_z_bar_self_duality)
    relations = [relation for relation in relations if relation != 0]
    relations += [
        normalize_conjugates(sp.diff(relation, coordinates[index]))
        for relation in relations
        for index in coordinates
    ]
    relations += list(sphere_self_duality)
    return tuple(numerator(relation) for relation in relations if relation != 0)

# ===============   Computing Yang-Mills residuals   ================

def commutator(left, right):
    return left * right - right * left

def derivative(matrix, direction):
    return matrix.applyfunc(lambda entry: normalize_conjugates(sp.diff(normalize_conjugates(entry), coordinates[direction])))

def compute_curvature(connection, index1, index2):
    return (
          derivative(connection[index2], index1)
        - derivative(connection[index1], index2)
        + commutator(connection[index1], connection[index2])
    )

def compute_raised_curvature(connection, metric="product"):
    """ Returns F^μν as a function of the indices """
    raised_curvature_cache = {}
    def raised_curvature(index1, index2):
        if (index1, index2) in raised_curvature_cache:
            return raised_curvature_cache[(index1, index2)]
        if (index2, index1) in raised_curvature_cache:
            return -raised_curvature_cache[(index2, index1)]

        # rather than sum over dummy indices, just pick out the conjugate coordinates
        # since that gives the only non-zero metric component
        dummy_index1 = conjugate_coordinate[index1]
        dummy_index2 = conjugate_coordinate[index2]

        F = compute_curvature(connection, dummy_index1, dummy_index2)
        F_up = inv_g(index1, dummy_index1, metric) * inv_g(index2, dummy_index2, metric) * F
        raised_curvature_cache[(index1, index2)] = F_up
        return raised_curvature_cache[(index1, index2)]
    return raised_curvature

def covariant_derivative(raised_curvature, index1, index2, index3, metric="product"):
    """ Returns ∇_λ F^μν for the chosen Levi-Civita connection """
    nabla_F = derivative(raised_curvature(index2, index3), index1)
    for dummy_index in coordinates:
        nabla_F += (
              christoffel(index2, index1, dummy_index, metric) * raised_curvature(dummy_index, index3)
            + christoffel(index3, index1, dummy_index, metric) * raised_curvature(index2, dummy_index)
        )
    return nabla_F

def gauge_covariant_derivative(connection, raised_curvature, index1, index2, index3, metric="product"):
    """ Returns D_λ F^μν """
    nabla_F = covariant_derivative(raised_curvature, index1, index2, index3, metric)
    A_wedge_F = commutator(connection[index1], raised_curvature(index2, index3))
    return nabla_F + A_wedge_F

def ym_tensor(connection, index, metric="product"):
    """ Returns D_μ F^μν """
    raised_curvature = compute_raised_curvature(connection, metric)
    return sum([
        gauge_covariant_derivative(connection, raised_curvature, dummy_index, dummy_index, index, metric)
        for dummy_index in coordinates
    ], sp.zeros(matrix_size)).applyfunc(sp.simplify)

def degen_ym_tensor(connection, index, relations=None):
    """ Returns D_i F^ai or D_a F^ai
        where a and i are the sphere and disk indices respectively
    """
    raised_curvature = compute_raised_curvature(connection, "product")
    restricted_coordinates = ["w", "w_bar"] if index in ["z", "z_bar"] else ["z", "z_bar"]
    residual = sum([
        gauge_covariant_derivative(connection, raised_curvature, dummy_index, dummy_index, index, "product")
        for dummy_index in restricted_coordinates
    ], sp.zeros(matrix_size))
    return impose_relations(residual, relations)

# ===============   main   ================

if __name__ == "__main__":
    start_time = perf_counter()
    try:
        main()
    finally:
        print(f"Finished in {perf_counter() - start_time:.2f} seconds")
