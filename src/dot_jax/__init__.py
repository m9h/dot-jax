"""dot-jax: JAX/Equinox toolbox for Diffuse Optical Tomography and fNIRS.

A differentiable, GPU-accelerated reimplementation of redbirdpy using the
JAX ecosystem (Equinox, Lineax, Optimistix). Enables autodiff Jacobians,
JIT compilation, and vmap over sources/detectors/wavelengths.
"""

__version__ = "0.1.0"

from ._types import C0, R_C0, ForwardResult, ReconResult
from .assembly import assemble_stiffness, assemble_mass, assemble_boundary, assemble_system_cw
from .forward import forward_cw, forward_cw_sparse, locate_sources, assemble_rhs, get_detector_values
from .spectral import spectral_forward_cw, compute_jacobian_mua
from .recon import reconstruct_mua
from .io import read_snirf, load_bids_nirs, snirf_to_dot_jax, SnirfData, BidsNirsRun
