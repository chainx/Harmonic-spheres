# Harmonic spheres

Numerical and symbolic experiments relating Yang–Mills connections on $S^4$
to harmonic maps from $S^2$ into a based loop group.

The code reconstructs the $SO(3)$-equivariant Yang–Mills solutions of Sadun
and Segert, expresses them in $S^2\times D^2$ coordinates, constructs the
Jarvis–Norbury frame on each disk, and tests the harmonic-map equation in a
truncated Fourier model. The BPST instanton is included as a reference case.

This is research code: numerical resolutions and tolerances are set directly
in the scripts, and the longer computations can be expensive.

## Mathematical outline

Write $w=u+iv$ for the coordinate on $S^2$ and $z=x+iy$ for the disk
coordinate. Given a connection $A$, the numerical pipeline:

1. pulls $A$ back to the disk $D_w$;
2. solves $D_{\bar z}g=0$ using Chebyshev–Lobatto collocation in the radial
   direction and Fourier collocation in the angular direction;
3. applies a numerical Iwasawa factorisation so that the boundary frame is
   unitary and normalized at $z=1$;
4. computes sphere derivatives of the normalized frame; and
5. evaluates the Grassmannian projector equation
   $[P,P_{uu}+P_{vv}]=0$ in a truncated loop-space representation.

The frame construction follows the approach of Jarvis and Norbury:

> S. Jarvis and P. Norbury, “Degenerating Metrics and Instantons on the
> Four-Sphere,” *Journal of Geometry and Physics* 27 (1998), 79–98.
> [doi:10.1016/S0393-0440(97)00067-3](https://doi.org/10.1016/S0393-0440(97)00067-3)

The Sadun–Segert connection is reconstructed from a Fourier approximation to
the reduced function $a_3(\theta)$, following:

> L. Sadun and J. Segert, “Stationary Points of the Yang–Mills Action,”
> *Communications on Pure and Applied Mathematics* 45 (1992).
> [doi:10.1002/cpa.3160450405](https://doi.org/10.1002/cpa.3160450405)

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- SymPy (only for the scripts in `symbolic manipulation/`)

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy sympy
```

Run commands from the repository root. Several modules resolve the `results/`
directory relative to the current working directory.

## Quick start

The repository already contains Fourier coefficients for the commonly used
Sadun–Segert cases $(r,t)=(-1,3)$ and $(-5,3)$, with $l=4$.

Construct a Jarvis–Norbury frame for the default $(-5,3)$ solution:

```bash
python3 jarvis_norbury_frame.py
```

Compute first and second sphere derivatives using the BPST reference
connection:

```bash
python3 construct_jn_frame_derivatives.py
```

Evaluate the truncated loop-group harmonicity residual:

```bash
python3 check_loop_group_harmonicity.py
```

Run the broader grid- and step-size diagnostics:

```bash
python3 diagnose_jn_harmonicity_error_sources.py
```

These frame and harmonicity calculations print PDE, boundary-unitarity, and
projector residuals. Their default collocation grids may take some time.

## Reconstructing a Sadun–Segert solution

The reduced Yang–Mills solver minimizes the symmetry-reduced action and saves
the resulting Fourier coefficients under `results/`:

```bash
python3 construct_sadun_segert_ym_solution.py
```

With the defaults this writes:

```text
results/r=-5_t=3_solution_with_28_modes.npy
```

The saved coefficients are consumed by
`load_sadun_segert_connection(l=4, r=-5, t=3)`. To use another $l,r,t$
combination, first generate the corresponding coefficient file and ensure its
name follows the same convention.

To inspect the reconstructed connection at representative points:

```bash
python3 construct_sadun_segert_in_wz_coords.py
```

To locate the two singular-orbit points in a fixed disk:

```bash
python3 find_sadun_segert_singular_points.py "0.2+0.3j"
```

## Verification and symbolic checks

Numerical consistency checks:

```bash
python3 consistency_checks/verify_sadun_segert_construction.py
python3 consistency_checks/verify_ym_in_wz_coords.py
```

Symbolic checks:

```bash
python3 "symbolic manipulation/verify_s4_coordinate_change.py"
python3 "symbolic manipulation/calculate_ym_tensor.py"
```

The Yang–Mills verification samples many points and uses finite differences,
so it is substantially slower than a unit test. The symbolic Yang–Mills
calculation may also be computationally intensive.

## Repository layout

| Path | Purpose |
| --- | --- |
| `construct_sadun_segert_ym_solution.py` | Solve the reduced Sadun–Segert variational problem and save Fourier coefficients |
| `construct_sadun_segert_in_wz_coords.py` | Reconstruct the connection in $(u,v,x,y)$ coordinates |
| `construct_bpst_in_wz_coords.py` | Construct the BPST reference connection in the same coordinates |
| `disk_solve.py` | Shared disk grids, differentiation matrices, normalization, and diagnostics |
| `jarvis_norbury_frame.py` | Solve $D_{\bar z}g=0$ and construct the normalized frame |
| `iwasawa_factorisation.py` | Numerical loop Iwasawa factorisation |
| `construct_jn_frame_derivatives.py` | Compute sphere derivatives and Maurer–Cartan data |
| `check_loop_group_harmonicity.py` | Evaluate the truncated Grassmannian projector residual |
| `diagnose_jn_harmonicity_error_sources.py` | Study grid, finite-difference, and truncation sensitivity |
| `find_sadun_segert_singular_points.py` | Locate singular-orbit points for a specified $w$ |
| `consistency_checks/` | Numerical reconstruction and Yang–Mills checks |
| `symbolic manipulation/` | SymPy coordinate and tensor identities |
| `results/` | Saved Fourier coefficients and numerical frame data |

`construct_jn_frame_derivatives_old.py` is a legacy implementation retained
for comparison; the current implementation is
`construct_jn_frame_derivatives.py`.

## Using the code as modules

The principal constructors return callables or solver objects and can be used
from a Python session:

```python
from construct_sadun_segert_in_wz_coords import load_sadun_segert_connection
from jarvis_norbury_frame import JN_frame_solver

A_mu = load_sadun_segert_connection(l=4, r=-5, t=3)
frame, eta_inverse = JN_frame_solver(A_mu).construct_JN_frame(
    w=0.2 + 0.3j
)
```

The connection callable accepts `[u, v, x, y]` and returns the four
anti-Hermitian $2\times2$ matrices $[A_u,A_v,A_x,A_y]$.

## License

This project is available under the [MIT License](LICENSE).
