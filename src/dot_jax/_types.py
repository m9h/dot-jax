"""Core types, constants, and result containers for dot-jax."""

from typing import NamedTuple
import jax.numpy as jnp

# Speed of light in mm/s
C0 = 299792458000.0
R_C0 = 1.0 / C0


class ForwardResult(NamedTuple):
    """Result of a forward solve."""
    detval: jnp.ndarray  # (n_det, n_src) detector measurements
    phi: jnp.ndarray     # (n_nodes, n_cols) fluence field


class ReconResult(NamedTuple):
    """Result of a reconstruction."""
    mua: jnp.ndarray        # recovered absorption coefficient
    musp: jnp.ndarray       # recovered reduced scattering coefficient
    residuals: jnp.ndarray  # residual per iteration
