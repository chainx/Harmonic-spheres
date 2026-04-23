import numpy as np
from scipy.optimize import root

def _shift(point, index, step):
    shifted = point.copy()
    shifted[index] += step
    return shifted

def _mixed_partial(function, point, row, row_step, col, col_step):
    total = 0.0
    for row_sign in (1.0, -1.0):
        for col_sign in (1.0, -1.0):
            shifted = point.copy()
            shifted[row] += row_sign * row_step
            shifted[col] += col_sign * col_step
            total += row_sign * col_sign * function(shifted)
    return total / (4 * row_step * col_step)

def finite_difference_gradient(function, point, *, step_scale=1e-5):
    steps = step_scale * (1 + np.abs(point))
    gradient = np.empty(point.size)

    for i, step in enumerate(steps):
        coarse = (function(_shift(point, i, step)) - function(_shift(point, i, -step))) / (2 * step)
        half_step = 0.5 * step
        fine = (function(_shift(point, i, half_step)) - function(_shift(point, i, -half_step))) / (2 * half_step)
        gradient[i] = (4 * fine - coarse) / 3

    return gradient

def finite_difference_hessian(function, point, *, hessian_step_scale=1e-5):
    steps = hessian_step_scale * (1 + np.abs(point))
    hessian = np.zeros((point.size, point.size))
    base_value = function(point)

    for row, row_step in enumerate(steps):
        coarse = (function(_shift(point, row, row_step)) - 2 * base_value + function(_shift(point, row, -row_step))) / row_step**2
        half_step = 0.5 * row_step
        fine = (function(_shift(point, row, half_step)) - 2 * base_value + function(_shift(point, row, -half_step))) / half_step**2
        hessian[row, row] = (4 * fine - coarse) / 3

        for col in range(row + 1, point.size):
            col_step = steps[col]
            coarse = _mixed_partial(function, point, row, row_step, col, col_step)
            fine = _mixed_partial(function, point, row, 0.5 * row_step, col, 0.5 * col_step)
            mixed = (4 * fine - coarse) / 3
            hessian[row, col] = mixed
            hessian[col, row] = mixed

    return hessian

def minimise(
    function,
    initial_point,
    args=(),
    *,
    gradient_tol=1e-15,
    max_steps=8,
    hessian_regularization=1e-10,
    min_step_scale=1e-6,
):
    point = np.array(initial_point, copy=True)
    function_with_args = lambda x: function(x, *args)
    gradient_with_args = lambda x: finite_difference_gradient(function_with_args, x)
    hessian_with_args = lambda x: finite_difference_hessian(function_with_args, x)

    root_result = root(gradient_with_args, point, jac=hessian_with_args, method="hybr", options={"xtol": gradient_tol})
    if root_result.success and function_with_args(root_result.x) <= function_with_args(point):
        point = root_result.x

    for _ in range(max_steps):
        gradient_value = gradient_with_args(point)
        if np.linalg.norm(gradient_value) < gradient_tol:
            return point

        hessian_value = hessian_with_args(point)
        for regularization_power in range(6):
            regularization = hessian_regularization * (10 ** regularization_power)
            try:
                direction = np.linalg.solve(hessian_value + regularization * np.eye(point.size), -gradient_value)
                break
            except np.linalg.LinAlgError:
                if regularization_power == 5:
                    return point
        base_value = function_with_args(point)
        step_scale = 1.0

        while step_scale >= min_step_scale:
            trial_point = point + step_scale * direction
            if function_with_args(trial_point) < base_value:
                point = trial_point
                break
            step_scale *= 0.5
        else:
            return point

    return point
