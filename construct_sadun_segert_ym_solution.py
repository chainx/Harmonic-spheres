"""Constructs Sadun/Segert SO(3)-equivariant Yang-Mills solutions on S^4.

This file constructs the reduced solution by finding stationary points
of the reduced Yang-Mills action for connections A of the form

    A(θ, g) = \sum_{i=1}^3 -a_i(θ) (g^{-1}dg)^i T_i

where θ ∈ [0, π/3] and g ∈ SO(3) with T_i the so(3) basis vectors. 
This connection is defined on a trivial bundle over the base space [0, π/3] x SO(3), however boundary conditions
are imposed on a_i so that the connection descends to a smooth connection on a principal bundle over S^4.
More details on how we extract the connection on S^4 from this are in convert_to_wz_coords.py.




First, a_3 is extended to a function from [0, 2π], and then a_2 and a_1 are extracted via

    a_2(θ) = a_3(θ + 2π/3)     a_1(θ) = a_3(θ - 2π/3)

The boundary conditions defined by (r,t) are expressed in terms of a_3 are

    a_3(0) = r     a_3(π) = t

and when r,t ≠ 1 there are additional boundary conditions

    a_3(π/3) = 0 = a_3(2π/3)

These solutions are smooth on S^4 iff r+1 and t+1 are divisible by 4.




We express a_3 in terms of Fourier coefficients A_n which satisfy A_n = A_n^* = A_{-n}.
Then we truncate with the parameter l, keeping only the first 2l+6 positive fourier modes and the corresponding negative modes.
The boundary conditions are solved by constraining the first four Fourier modes in terms of the remaining 2l+2 modes.
These 2l+2 modes are the free_params from which the remaining modes are constructed in fourier_coeff_from_free_params.

the free_params defining a_3 are then obtained via minimising the symmetry reduced action

    S = π^2 ∫ d0 a'_1^2 G1 + a'_2^2 G2 + a'_3^2 G3 + b1^2 / G1 + b2**2 / G2 + b3**2 / G3

defined in equation (2.9) of [1].

[1] Stationary Points of the Yang-Mills Action, by Sadun and Segert
Published in "Communications on Pure and Applied Mathematics" in 1992
https://doi.org/10.1002/cpa.3160450405 
"""

import argparse
from pathlib import Path
RESULTS_PATH = Path.cwd() / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

import numpy as np
from scipy.integrate import quad
from numerical_help.damped_newton import minimise

π = np.longdouble(np.pi)

L = 4 # Truncates to keep the first 2l + 6 positive and negative Fourier modes
R = -5; T = 3 # Inputs defining boundary conditions which uniquely define a_3(θ)

INTEGRATION_PARAMS = {
    "epsabs": 1e-12, # Absolute tolerance
    "epsrel": 1e-12, # Relative tolerance
    "limit": 20000,  # Max number of subdivisions of the domain
}
MINIMISATION_PARAMS = {
    "gradient_tol": 1e-15,
    "max_steps": 8,
    "hessian_regularization": 1e-10,
    "min_step_scale": 1e-6,
}

def main():
    args = parse_args()
    apply_args(args)
    
    fourier_coeff = construct_a3_fourier_coeff(L, R, T)
    np.save(RESULTS_PATH/f"r={R}_t={T}_solution_with_{2*(2*L+6)}_modes.npy", fourier_coeff)
    
    a3, _ = construct_a3_and_da3_from_fourier_coeff(fourier_coeff) # Reconstruct a_3(θ)
    verify_boundary_conditions(a3, R, T) # Verify that it satisfies the (r,t) boundary conditions

def construct_a3_fourier_coeff(l, r, t):
    initial_free_params = np.zeros(6*l + 2) # The initial guess is to set all the higher Fourier modes to 0

    # Find the free_params that minimize the action
    free_params = minimise(action, initial_free_params, args=(l, r, t), **MINIMISATION_PARAMS)

    return fourier_coeff_from_free_params(free_params, l, r, t) # Return the Fourier coefficients


# =============================================================================================


def fourier_coeff_from_free_params(free_params, l ,r ,t):
    """ Solve for first four Fourier coefficients using boundary conditions """

    fourier_coeff = np.zeros(6*l + 6)
    fourier_coeff[4:] = free_params

    fourier_coeff[0] = (r+t)/6 -2*np.sum(fourier_coeff[6::6])
    fourier_coeff[1] = (r-t)/6  - np.sum(fourier_coeff[7::6]) - np.sum(fourier_coeff[5::6])
    fourier_coeff[2] = (r+t)/6  - np.sum(fourier_coeff[8::6]) - np.sum(fourier_coeff[4::6])
    fourier_coeff[3] = (r-t)/12 - np.sum(fourier_coeff[9::6])

    return fourier_coeff

def construct_a3_and_da3_from_fourier_coeff(fourier_coeff):
    """ Constructs a_3(θ) and (da_3 / dθ)(θ) from the Fourier coefficients """
    def a3(θ):
        total = fourier_coeff[0]
        for n, coeff in enumerate(fourier_coeff[1:], start=1):
            total += 2 * coeff * np.cos(n * θ)
        return total
    def da3(θ):
        total = 0
        for n, coeff in enumerate(fourier_coeff[1:], start=1):
            total -= 2 * n * coeff * np.sin(n * θ)
        return total

    return a3, da3

def verify_boundary_conditions(a3, r, t):
    """ We are assuming r,t ≠ 1, hence the need for the final pair of boundary conditions """
    assert np.isclose(a3(0)    , r)
    assert np.isclose(a3(π)    , t)
    assert np.isclose(a3(π/3)  , 0)
    assert np.isclose(a3(2*π/3), 0)

def action(free_params, l, r, t):
    """ See equation (2.9) of [1] """
    fourier_coeff = fourier_coeff_from_free_params(free_params, l, r, t)
    a3, da3 = construct_a3_and_da3_from_fourier_coeff(fourier_coeff)

    def integrand(θ):
        f1 = 2*np.sin(π/3 + θ)
        f2 = 2*np.sin(π/3 - θ)
        f3 = 2*np.sin(θ)
        G1 = f2*f3/f1
        G2 = f1*f3/f2
        G3 = f1*f2/f3

        a1 = a3(θ - 2*π/3); da1 = da3(θ - 2*π/3)
        a2 = a3(θ + 2*π/3); da2 = da3(θ + 2*π/3)        

        b1 = a1 + a2*a3(θ)
        b2 = a2 + a3(θ)*a1
        b3 = a3(θ) + a1*a2

        return da1**2 * G1 + da2**2 * G2 + da3(θ)**2 * G3 + b1**2 / G1 + b2**2 / G2 + b3**2 / G3

    S, _ = quad(integrand, 0, π/3, **INTEGRATION_PARAMS)
    return S


# =============================================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Construct a Sadun-Segert Yang-Mills solution.")
    parser.add_argument("--l", type=int, default=4, help="Truncation parameter; keeps 2l + 6 positive Fourier modes.")
    parser.add_argument("--r", type=int, default=-5, help="Boundary condition a_3(0) = r.")
    parser.add_argument("--t", type=int, default=3, help="Boundary condition a_3(pi) = t.")
    parser.add_argument("--epsabs", type=float, default=INTEGRATION_PARAMS["epsabs"], help="Absolute tolerance for quad.")
    parser.add_argument("--epsrel", type=float, default=INTEGRATION_PARAMS["epsrel"], help="Relative tolerance for quad.")
    parser.add_argument("--limit", type=int, default=INTEGRATION_PARAMS["limit"], help="Maximum number of quad subdivisions.")
    parser.add_argument("--gradient-tol", type=float, default=MINIMISATION_PARAMS["gradient_tol"], help="Gradient tolerance for damped Newton minimisation.")
    parser.add_argument("--max-steps", type=int, default=MINIMISATION_PARAMS["max_steps"], help="Maximum number of damped Newton steps.")
    parser.add_argument("--hessian-regularization", type=float, default=MINIMISATION_PARAMS["hessian_regularization"], help="Initial Hessian regularization.")
    parser.add_argument("--min-step-scale", type=float, default=MINIMISATION_PARAMS["min_step_scale"], help="Minimum backtracking step scale.")
    return parser.parse_args()

def apply_args(args):
    L = args.l
    R = args.r ; T = args.t

    INTEGRATION_PARAMS.update({
        "epsabs": args.epsabs, 
        "epsrel": args.epsrel, 
        "limit": args.limit}
    )
    MINIMISATION_PARAMS.update({
        "gradient_tol": args.gradient_tol,
        "max_steps": args.max_steps,
        "hessian_regularization": args.hessian_regularization,
        "min_step_scale": args.min_step_scale,
    })

if __name__ == "__main__":
    main()
