"""Core types, constants, and result containers for dot-jax."""

from typing import NamedTuple
import jax.numpy as jnp

# Speed of light in mm/s
C0 = 299792458000.0
R_C0 = 1.0 / C0


class ForwardResult(NamedTuple):
    """Result of a CW forward solve.

    Attributes
    ----------
    detval : jnp.ndarray, shape (n_det, n_src)
        Detector measurement values.
    phi : jnp.ndarray, shape (nn, n_src)
        Fluence field at all mesh nodes for each source.
    """
    detval: jnp.ndarray  # (n_det, n_src) detector measurements
    phi: jnp.ndarray     # (n_nodes, n_cols) fluence field


class ReconResult(NamedTuple):
    """Result of an image reconstruction.

    Attributes
    ----------
    mua : jnp.ndarray, shape (nn,)
        Recovered nodal absorption coefficient (1/mm).
    musp : jnp.ndarray, shape (nn,)
        Recovered nodal reduced scattering coefficient (1/mm).
    residuals : jnp.ndarray, shape (n_steps + 1,)
        Data-model residual norm at each Gauss-Newton iteration.
    """
    mua: jnp.ndarray        # recovered absorption coefficient
    musp: jnp.ndarray       # recovered reduced scattering coefficient
    residuals: jnp.ndarray  # residual per iteration
