"""dot-jax: JAX/Equinox toolbox for Diffuse Optical Tomography and fNIRS.

A differentiable, GPU-accelerated reimplementation of redbirdpy using the
JAX ecosystem (Equinox, Lineax, Optimistix). Enables autodiff Jacobians,
JIT compilation, and vmap over sources/detectors/wavelengths.
"""

__version__ = "0.1.0"

from ._types import C0, R_C0, ForwardResult, ReconResult
