# dot-jax

**JAX/Equinox toolbox for Diffuse Optical Tomography (DOT) and fNIRS.**

A differentiable, GPU-accelerated reimplementation of [redbirdpy](https://github.com/fangq/redbirdpy) using the JAX ecosystem ([Equinox](https://github.com/patrick-kidger/equinox), [Lineax](https://github.com/patrick-kidger/lineax), [Optimistix](https://github.com/patrick-kidger/optimistix)).

dot-jax provides the complete pipeline for CW diffuse optical tomography: analytical solutions, chromophore spectroscopy, FEM mesh handling, system matrix assembly, forward solving, and image reconstruction — all composable with `jax.jit`, `jax.grad`, and `jax.vmap`.

## Features

- **Autodiff Jacobians** — differentiate through the full forward solve w.r.t. optical properties
- **JIT compilation** — all core functions compile to XLA for CPU/GPU
- **vmap** — batch over sources, detectors, or wavelengths
- **Equinox modules** — `FEMMesh` is a proper JAX pytree
- **Lineax solvers** — sparse-ready linear system solving with autodiff support
- **Cross-validated** — tested against redbirdpy and scipy at every layer

## Installation

```bash
pip install -e ".[test]"
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e ".[test]"
```

### Dependencies

- JAX >= 0.4.30
- Equinox >= 0.11.0
- Lineax >= 0.0.5
- jaxtyping >= 0.2.24
- NumPy >= 1.26

Optional: `scipy` and `redbirdpy` for cross-validation tests.

## Quick Start

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.forward import forward_cw

# Build a tetrahedral mesh
node = jnp.array([
    [0, 0, 0], [20, 0, 0], [0, 20, 0], [20, 20, 0],
    [0, 0, 20], [20, 0, 20], [0, 20, 20], [20, 20, 20],
], dtype=jnp.float64)
elem = jnp.array([
    [0,1,2,4], [1,2,3,7], [1,2,4,7], [1,4,5,7], [2,4,6,7],
], dtype=jnp.int32)
mesh = FEMMesh.create(node, elem)

# Forward solve
srcpos = jnp.array([[5.0, 5.0, 5.0]])
detpos = jnp.array([[15.0, 15.0, 15.0]])
result = forward_cw(mesh, mua=0.01, musp=1.0, srcpos=srcpos, detpos=detpos)

print(f"Detector value: {float(result.detval[0,0]):.4e}")

# Autodiff: sensitivity of detector signal to absorption
grad_mua = jax.grad(lambda mua: jnp.sum(
    forward_cw(mesh, mua, 1.0, srcpos, detpos).detval
))(0.01)
print(f"d(signal)/d(mua) = {float(grad_mua):.4e}")
```

## Module Reference

| Module | Description |
|--------|-------------|
| `analytical` | Closed-form CW solutions (infinite/semi-infinite media), spherical Bessel/Hankel functions, spherical harmonics |
| `property` | Chromophore extinction tables (HbO2, Hb, water, lipids, aa3), Beer-Lambert absorption, scattering power law |
| `mesh` | `FEMMesh` Equinox module with precomputed operators (deldotdel, element/nodal volumes, surface extraction) |
| `assembly` | FEM system matrix A = K + M + C (stiffness, mass, Robin BC) |
| `forward` | CW forward solver via Lineax with barycentric source placement and adjoint detector extraction |
| `spectral` | Multi-wavelength forward model and adjoint Jacobian computation |
| `recon` | Gauss-Newton image reconstruction with Tikhonov/Levenberg-Marquardt regularisation |

## Examples

Runnable tutorials are in `examples/`:

| Example | What it demonstrates |
|---------|---------------------|
| `01_analytical_solutions.py` | CW fluence in infinite and semi-infinite media, autodiff, spherical symmetry |
| `02_optical_properties.py` | Chromophore spectra, Beer-Lambert law, scattering power law, isosbestic point |
| `03_fem_forward_solve.py` | Mesh construction, system matrix assembly, forward solve, Jacobian computation |
| `04_image_reconstruction.py` | Synthetic data generation, linearised reconstruction, residual convergence |

## Research Background

dot-jax builds on several decades of work in biomedical optics, diffuse optical tomography, and functional near-infrared spectroscopy.

### Photon transport in tissue

Near-infrared light (650--950 nm) penetrates several centimetres into biological tissue, scattered primarily by cell membranes and mitochondria, and absorbed by hemoglobin, water, and lipids. The radiative transfer equation (RTE) governs photon propagation, but the **diffusion approximation** — valid when scattering dominates absorption — reduces it to a tractable PDE:

$$-\nabla \cdot (D \, \nabla \Phi) + \mu_a \, \Phi = S$$

where $\Phi$ is the photon fluence rate, $D = 1/[3(\mu_a + \mu_s')]$ is the diffusion coefficient, $\mu_a$ is the absorption coefficient, and $\mu_s'$ is the reduced scattering coefficient. This approximation was established by **Ishimaru (1978)** and formalised for tissue optics by **Patterson, Chance, and Wilson (1989)**.

### Analytical solutions

Closed-form Green's functions exist for homogeneous geometries:
- **Infinite medium**: the point-source kernel $\Phi(r) = \exp(-\mu_{\text{eff}} r) / (4\pi D r)$
- **Semi-infinite medium**: the image-source method with extrapolated boundary conditions (**Farrell, Patterson & Wilson, 1992; Haskell et al., 1994**)

These are the foundations of the `analytical` module and remain important for model validation.

### Finite element methods for DOT

Real tissue geometries require numerical methods. The FEM approach was introduced to biomedical optics by **Arridge, Schweiger, Hiraoka & Delpy (1993)** and **Paulsen & Jiang (1995)**. The key insight is that the diffusion equation maps naturally to a symmetric positive-definite linear system whose assembly parallels structural mechanics — but with diffusion and absorption replacing elasticity.

The **TOAST** (Temporal Optical Absorption and Scattering Tomography) package by **Schweiger & Arridge** and the **NIRFAST** package by **Dehghani et al. (2009)** established the standard FEM-based DOT pipeline that redbirdpy and dot-jax follow: assemble K + M + C, solve with direct or iterative methods, extract detector values via the adjoint.

### Chromophore spectroscopy

The wavelength dependence of tissue absorption reveals chromophore concentrations via the **modified Beer-Lambert law**. Hemoglobin extinction data compiled by **Scott Prahl** at the Oregon Medical Laser Center (OMLC) — tracing back to measurements by **Takatani & Graham (1979)** and **Zijlstra, Buursma & Meeuwsen-van der Roest (1991)** — provides the spectroscopic basis for fNIRS.

The **isosbestic point** near 800 nm, where oxy- and deoxyhemoglobin have equal extinction, is a key design constraint for fNIRS instruments. Multi-wavelength measurements (typically 690 and 830 nm) enable separation of HbO2 and Hb changes.

### Image reconstruction

DOT image reconstruction is the inverse of the forward problem: given boundary measurements, recover the spatial distribution of optical properties. This is a severely ill-posed nonlinear inverse problem. The standard approach, reviewed comprehensively by **Arridge (1999)**, uses:

1. **Linearisation** via the Born or Rytov approximation
2. The **adjoint Jacobian** $J_{d,s,n} = -\Phi_s(n) \cdot \Phi_d(n) \cdot V_n$ (the sensitivity of measurement $d,s$ to absorption at node $n$)
3. **Tikhonov regularisation** to stabilise the inversion

dot-jax implements this with a twist: because the forward solve is fully differentiable via JAX, the Jacobian can also be computed via automatic differentiation — providing an independent verification of the adjoint formula.

### Functional near-infrared spectroscopy (fNIRS)

fNIRS, pioneered by **Jöbsis (1977)**, uses the same physics in a simpler configuration: surface-mounted sources and detectors measure changes in light attenuation that reflect hemodynamic activity in the brain. The DOT forward model provides the physical foundation for fNIRS signal interpretation, while the Beer-Lambert law connects optical changes to oxy/deoxyhemoglobin concentration changes.

### The redbirdpy lineage

dot-jax is a direct reimplementation of **redbirdpy** by Qianqian Fang, which is itself a Python port of the MATLAB **redbird** toolbox. Fang's broader ecosystem includes [MCX](http://mcx.space/) (Monte Carlo eXtreme) for stochastic photon transport and [iso2mesh](http://iso2mesh.sf.net/) for tetrahedral mesh generation. dot-jax preserves the mathematical formulations and cross-validates against redbirdpy at every layer while adding:

- **Automatic differentiation** through the entire forward/inverse pipeline
- **JIT compilation** for performance on CPU and GPU
- **Functional composition** via JAX transformations (vmap, grad, jit)
- **Equinox modules** as idiomatic JAX data structures

### Key references

1. A. Ishimaru, *Wave Propagation and Scattering in Random Media*, Academic Press, 1978.
2. F. F. Jöbsis, "Noninvasive, infrared monitoring of cerebral and myocardial oxygen sufficiency and circulatory parameters," *Science*, 198(4323):1264--1267, 1977.
3. T. J. Farrell, M. S. Patterson, and B. C. Wilson, "A diffusion theory model of spatially resolved, steady-state diffuse reflectance," *Med. Phys.*, 19(4):879--888, 1992.
4. S. R. Arridge, M. Schweiger, M. Hiraoka, and D. T. Delpy, "A finite element approach for modeling photon transport in tissue," *Med. Phys.*, 20(2):299--309, 1993.
5. R. C. Haskell et al., "Boundary conditions for the diffusion equation in radiative transfer," *JOSA A*, 11(10):2727--2741, 1994.
6. S. R. Arridge, "Optical tomography in medical imaging," *Inverse Problems*, 15(2):R41--R93, 1999.
7. S. Prahl, "Optical absorption of hemoglobin," Oregon Medical Laser Center, https://omlc.org/spectra/hemoglobin/, 1999.
8. D. A. Boas et al., "Imaging the body with diffuse optical tomography," *IEEE Signal Processing Magazine*, 18(6):57--75, 2001.
9. H. Dehghani et al., "Near infrared optical tomography using NIRFAST," *Int. J. Numer. Methods Biomed. Eng.*, 25(6):711--732, 2009.
10. Q. Fang, "Mesh-based Monte Carlo method using fast ray-tracing in Plucker coordinates," *Biomed. Opt. Express*, 1(1):165--175, 2010.

## Testing

```bash
python -m pytest tests/ -v
```

169 tests covering:
- Mathematical properties (symmetry, positivity, conservation)
- Known-value validation against analytical formulas
- Cross-validation against redbirdpy and scipy
- JIT/grad/vmap compatibility for all core functions

## License

GPL-3.0
