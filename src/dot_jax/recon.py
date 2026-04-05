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
    solve_dual: Dual-formulation regularised solve (Kernel/Holoscan style)
    compute_lcurve: L-curve for regularization parameter selection
    select_lambda_lcurve: Optimal lambda at L-curve corner
    select_lambda_gcv: Optimal lambda via Generalized Cross-Validation
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


def solve_dual(J, data, reg=0.01):
    """Dual-formulation regularised solve for underdetermined systems.

    Solves x = J^T (J J^T + lambda * sqrt(||JJ^T||) * I)^{-1} d

    This is more efficient than the primal form when n_nodes >> n_meas
    (underdetermined case typical in DOT). Based on the Kernel/NVIDIA
    Holoscan BCI reconstruction pipeline.

    Parameters
    ----------
    J : (n_meas, n_nodes) — Jacobian/sensitivity matrix.
    data : (n_meas,) — measurement vector.
    reg : float — regularization parameter.

    Returns
    -------
    x : (n_nodes,) — reconstructed parameter vector.
    """
    JJt = J @ J.T
    norm_scale = jnp.sqrt(jnp.linalg.norm(JJt))
    H_reg = JJt + reg * norm_scale * jnp.eye(J.shape[0])
    # Symmetrise for numerical stability
    H_reg = 0.5 * (H_reg + H_reg.T)
    alpha = jnp.linalg.solve(H_reg, data)
    return J.T @ alpha


# =============================================================================
# Regularization parameter selection
# =============================================================================


def compute_lcurve(J, data, lambdas):
    """Compute L-curve: residual norm vs solution norm for each lambda.

    Parameters
    ----------
    J : (n_meas, n_nodes) — Jacobian matrix.
    data : (n_meas,) — measurement vector.
    lambdas : (n_lambda,) — regularization parameters.

    Returns
    -------
    residual_norms : (n_lambda,) — ||J x - d||_2 for each lambda.
    solution_norms : (n_lambda,) — ||x||_2 for each lambda.
    """
    JtJ = J.T @ J
    Jtd = J.T @ data
    nn = J.shape[1]

    res_norms = []
    sol_norms = []
    for lam in lambdas:
        x = jnp.linalg.solve(JtJ + float(lam) * jnp.eye(nn), Jtd)
        res_norms.append(jnp.linalg.norm(J @ x - data))
        sol_norms.append(jnp.linalg.norm(x))

    return jnp.array(res_norms), jnp.array(sol_norms)


def _default_lambdas(J, n=50):
    """Generate default lambda range from singular values of J."""
    s = jnp.linalg.svd(J, compute_uv=False)
    s_pos = s[s > 1e-15]
    if len(s_pos) < 2:
        return jnp.logspace(-6, 2, n)
    smin, smax = float(s_pos[-1]), float(s_pos[0])
    return jnp.logspace(jnp.log10(smin ** 2), jnp.log10(smax ** 2), n)


def select_lambda_lcurve(J, data, lambdas=None):
    """Select optimal regularization parameter at L-curve corner.

    Finds the point of maximum curvature on the log-log L-curve.

    Parameters
    ----------
    J : (n_meas, n_nodes) — Jacobian.
    data : (n_meas,) — measurements.
    lambdas : (n_lambda,) optional — candidate values.

    Returns
    -------
    lambda_opt : float — optimal regularization parameter.
    """
    if lambdas is None:
        lambdas = _default_lambdas(J)

    res_norms, sol_norms = compute_lcurve(J, data, lambdas)

    # Maximum curvature on log-log L-curve
    x = jnp.log(res_norms + 1e-30)
    y = jnp.log(sol_norms + 1e-30)

    # Discrete curvature via second differences
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    d2x = jnp.diff(dx)
    d2y = jnp.diff(dy)

    # Curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    xp = (dx[:-1] + dx[1:]) / 2
    yp = (dy[:-1] + dy[1:]) / 2
    kappa = jnp.abs(xp * d2y - yp * d2x) / (xp ** 2 + yp ** 2) ** 1.5

    corner_idx = jnp.argmax(kappa) + 1  # +1 for the diff offset
    return float(lambdas[corner_idx])


def select_lambda_gcv(J, data, lambdas=None):
    """Select optimal regularization parameter via GCV.

    GCV(lambda) = ||Jx - d||^2 / trace(I - H)^2
    where H = J (J^T J + lambda I)^{-1} J^T.

    Parameters
    ----------
    J : (n_meas, n_nodes) — Jacobian.
    data : (n_meas,) — measurements.
    lambdas : (n_lambda,) optional — candidate values.

    Returns
    -------
    lambda_opt : float — optimal regularization parameter.
    """
    if lambdas is None:
        lambdas = _default_lambdas(J)

    m = J.shape[0]
    nn = J.shape[1]
    JtJ = J.T @ J
    Jtd = J.T @ data

    gcv_vals = []
    for lam in lambdas:
        A_inv = jnp.linalg.solve(JtJ + float(lam) * jnp.eye(nn), jnp.eye(nn))
        H = J @ A_inv @ J.T
        x = A_inv @ Jtd
        residual = J @ x - data
        denom = (jnp.trace(jnp.eye(m) - H) / m) ** 2
        gcv = jnp.sum(residual ** 2) / m / jnp.maximum(denom, 1e-30)
        gcv_vals.append(gcv)

    gcv_vals = jnp.array(gcv_vals)
    return float(lambdas[jnp.argmin(gcv_vals)])
