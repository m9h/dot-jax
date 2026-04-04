"""Gauss-Newton image reconstruction.

Linearised image reconstruction for nodal absorption coefficient
using the Born approximation with Tikhonov regularisation.

The adjoint Jacobian follows Arridge (1999) and the Levenberg-Marquardt
regularisation follows the standard formulation in Dehghani et al. (2009).

References
----------
.. [1] S. R. Arridge, "Optical tomography in medical imaging,"
       Inverse Problems, vol. 15, no. 2, pp. R41-R93, 1999.
.. [2] H. Dehghani et al., "Near infrared optical tomography using
       NIRFAST," Int. J. Numer. Methods Biomed. Eng., vol. 25,
       pp. 711-732, 2009.

Functions:
    reconstruct_mua: Linearised mua reconstruction from perturbation data
"""

import jax.numpy as jnp

from ._types import ReconResult
from .forward import forward_cw
from .spectral import compute_jacobian_mua


def reconstruct_mua(mesh, data, srcpos, detpos, mua0, musp,
                     n_in=1.37, n_out=1.0, max_steps=1,
                     reg_param=1e-4):
    """Linearised reconstruction of nodal absorption coefficient.

    Single-step (Born approximation) or iterative Gauss-Newton
    reconstruction. At each step::

        delta_mua = (J^T J + lambda * diag(J^T J))^{-1} J^T (data - pred)

    Parameters
    ----------
    mesh : FEMMesh
    data : (n_det, n_src) — measured detector values.
    srcpos : (n_src, 3) — source positions.
    detpos : (n_det, 3) — detector positions.
    mua0 : float — initial/background absorption coefficient.
    musp : float — reduced scattering coefficient (fixed).
    n_in, n_out : float — refractive indices.
    max_steps : int — number of Gauss-Newton iterations.
    reg_param : float — Tikhonov regularisation parameter.

    Returns
    -------
    ReconResult(mua, musp, residuals)
        mua : (nn,) — reconstructed nodal absorption.
        musp : (nn,) — unchanged scattering (returned for completeness).
        residuals : (max_steps + 1,) — residual norm per iteration.
    """
    nn = mesh.nn
    mua_current = mua0  # scalar operating point
    mua_node = jnp.full(nn, mua0)
    residuals = []

    for step in range(max_steps):
        # Forward solve at current operating point
        result = forward_cw(mesh, mua_current, musp, srcpos, detpos, n_in, n_out)
        predicted = result.detval

        # Residual
        diff = data - predicted
        res_norm = jnp.sqrt(jnp.sum(diff ** 2))
        residuals.append(res_norm)

        # Jacobian at current operating point
        J = compute_jacobian_mua(mesh, mua_current, musp, srcpos, detpos, n_in, n_out)

        # Gauss-Newton with Levenberg-Marquardt regularisation
        diff_flat = diff.ravel()
        JtJ = J.T @ J
        reg = reg_param * jnp.diag(jnp.diag(JtJ) + 1e-20)
        rhs = J.T @ diff_flat

        delta_mua = jnp.linalg.solve(JtJ + reg, rhs)
        mua_node = mua_node + delta_mua

        # Update operating point to mean of reconstructed field
        mua_current = jnp.clip(jnp.mean(mua_node), 1e-6, None)

    # Final residual
    result = forward_cw(mesh, mua_current, musp, srcpos, detpos, n_in, n_out)
    final_res = jnp.sqrt(jnp.sum((data - result.detval) ** 2))
    residuals.append(final_res)

    return ReconResult(
        mua=mua_node,
        musp=jnp.full(nn, musp),
        residuals=jnp.array(residuals),
    )
