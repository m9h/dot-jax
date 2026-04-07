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
from .io import (
    read_snirf, load_bids_nirs, snirf_to_dot_jax, SnirfData, BidsNirsRun,
    read_jmesh, fetch_neurojson, get_mcx_optical_properties, load_brain_mesh,
)
from .hemodynamics import (
    intensity_to_od, bandpass_filter, downsample, spectral_unmix,
    compute_channel_snr, prune_channels,
    detect_motion_artifacts, correct_motion_spline,
    normalize_od, zscore_images, compute_gvtd,
    identify_short_channels, regress_short_channels,
)
from .mesh import smooth_on_mesh
from .recon import compute_lcurve, select_lambda_lcurve, select_lambda_gcv, solve_dual, reconstruct_image
from .spectral import normalize_jacobian
from .td_forward import assemble_mass_time, td_source_pulse, td_forward_cw, dtof_moments
from .realtime import RealtimePipeline, EpochAccumulator
