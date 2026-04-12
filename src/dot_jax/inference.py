"""Physiologically-constrained DOT inference.

Couples dot-jax optical physics with vpjax vascular physiology to
produce temporally smooth, physically plausible HbO/HbR reconstructions.

The key idea (Diamond et al. 2006): instead of treating each fNIRS
frame independently, use the Balloon-Windkessel hemodynamic model as
a temporal prior. This enforces:
  - Smooth hemodynamic response dynamics (~6s rise time)
  - Anti-correlated HbO/HbR (both from the same vascular model)
  - Physiologically bounded concentrations

The implementation uses per-node Extended Kalman Filtering (EKF)
with JAX-native vectorization (vmap over nodes, lax.scan over time).

References
----------
.. [1] S. G. Diamond et al., "Dynamic physiological modeling for
       functional diffuse optical tomography," NeuroImage, 2006.
.. [2] R. E. Kalman, "A new approach to linear filtering and
       prediction problems," J. Basic Eng., 1960.

Functions:
    physiological_filter: EKF-based temporal smoothing of HbO/HbR
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class PhysiologicalResult(NamedTuple):
    """Result of physiological filtering.

    Attributes
    ----------
    hbo : (n_time, nn) — smoothed oxyhemoglobin changes (mM).
    hbr : (n_time, nn) — smoothed deoxyhemoglobin changes (mM).
    neural : (n_time, nn) — estimated neural activity per node.
    """
    hbo: np.ndarray
    hbr: np.ndarray
    neural: np.ndarray


# Baseline hemoglobin concentrations (mM)
_HbO_0 = 0.060
_HbR_0 = 0.040
_HbT_0 = _HbO_0 + _HbR_0


def _balloon_rhs(state, kappa, gamma, tau, alpha, E0):
    """Balloon-Windkessel ODE right-hand side (pure JAX).

    State: [u, s, f, v, q] where u is neural activity.
    """
    u = state[0]
    s = state[1]
    f = jnp.maximum(state[2], 0.01)
    v = jnp.maximum(state[3], 0.01)
    q = jnp.maximum(state[4], 0.01)

    du = 0.0  # neural activity: random walk (no dynamics)
    ds = u - kappa * s - gamma * (f - 1.0)
    df = s
    dv = (f - v ** (1.0 / alpha)) / tau
    E_f = 1.0 - (1.0 - E0) ** (1.0 / f)
    dq = (f * E_f / E0 - q * v ** (1.0 / alpha - 1.0)) / tau

    return jnp.array([du, ds, df, dv, dq])


def _discrete_step(x, dt, kappa, gamma, tau, alpha, E0):
    """One Euler step of the Balloon ODE (JIT-compatible)."""
    dx = _balloon_rhs(x, kappa, gamma, tau, alpha, E0)
    x_new = x + dt * dx
    # Soft clamp: keep f, v, q > 0.01 without breaking gradients
    x_new = x_new.at[2].set(jnp.maximum(x_new[2], 0.01))
    x_new = x_new.at[3].set(jnp.maximum(x_new[3], 0.01))
    x_new = x_new.at[4].set(jnp.maximum(x_new[4], 0.01))
    return x_new


def _observe(x):
    """Observation function: state → [delta_HbO, delta_HbR]."""
    v, q = x[3], x[4]
    dhbo = _HbT_0 * v - _HbR_0 * q - _HbO_0
    dhbr = _HbR_0 * q - _HbR_0
    return jnp.array([dhbo, dhbr])


def physiological_filter(hbo_observed, hbr_observed, dt, balloon_params,
                         process_noise=None, obs_noise=None):
    """EKF-based physiological filtering of DOT reconstructions.

    Per-node Extended Kalman Filter using Balloon-Windkessel dynamics
    as the temporal model and frame-by-frame HbO/HbR reconstructions
    as observations. Vectorized via jax.vmap over nodes.

    Parameters
    ----------
    hbo_observed : (n_time, nn) — noisy HbO reconstructions.
    hbr_observed : (n_time, nn) — noisy HbR reconstructions.
    dt : float — time step (seconds).
    balloon_params : vpjax BalloonParams.
    process_noise : (5,) optional — diagonal process noise variances.
    obs_noise : (2,) optional — observation noise variances.

    Returns
    -------
    PhysiologicalResult(hbo, hbr, neural)
    """
    hbo_obs = jnp.asarray(hbo_observed)
    hbr_obs = jnp.asarray(hbr_observed)
    n_time, nn = hbo_obs.shape

    # Detect units: if max > 1, assume micromolar → convert to mM
    # (Balloon model operates in mM; frame-by-frame recon typically gives uM)
    scale = jnp.where(jnp.max(jnp.abs(hbo_obs)) > 1.0, 1e-3, 1.0)
    hbo_obs = hbo_obs * scale
    hbr_obs = hbr_obs * scale

    # Extract scalar params for JIT
    kappa = float(balloon_params.kappa)
    gamma = float(balloon_params.gamma)
    tau = float(balloon_params.tau)
    alpha = float(balloon_params.alpha)
    E0 = float(balloon_params.E0)

    # Default process noise
    if process_noise is None:
        process_noise = jnp.array([1e-4, 1e-6, 1e-6, 1e-6, 1e-6])
    Q = jnp.diag(process_noise)

    # Default observation noise from data
    if obs_noise is None:
        hbo_dvar = jnp.var(jnp.diff(hbo_obs, axis=0), axis=0).mean()
        hbr_dvar = jnp.var(jnp.diff(hbr_obs, axis=0), axis=0).mean()
        obs_noise = jnp.array([jnp.maximum(hbo_dvar, 1e-10),
                               jnp.maximum(hbr_dvar, 1e-10)])
    R = jnp.diag(obs_noise)

    # Observation Jacobian (constant linearization around steady state)
    H = jnp.array([
        [0.0, 0.0, 0.0, _HbT_0, -_HbR_0],
        [0.0, 0.0, 0.0, 0.0,     _HbR_0],
    ])

    # Dynamics Jacobian via jax.jacfwd (computed once at steady state,
    # reused — first-order EKF approximation)
    x_ss = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
    F = jax.jacfwd(lambda x: _discrete_step(x, dt, kappa, gamma, tau, alpha, E0))(x_ss)

    # Stack observations: (n_time, 2, nn) → per-node (n_time, 2)
    y_all = jnp.stack([hbo_obs, hbr_obs], axis=1)  # (n_time, 2, nn)

    # EKF scan step (for a single node)
    def ekf_step(carry, y_t):
        x, P = carry

        # Predict
        x_pred = _discrete_step(x, dt, kappa, gamma, tau, alpha, E0)
        P_pred = F @ P @ F.T + Q

        # Update
        y_pred = _observe(x_pred)
        innovation = y_t - y_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ jnp.linalg.inv(S)

        x_upd = x_pred + K @ innovation
        P_upd = (jnp.eye(5) - K @ H) @ P_pred

        # Soft clamp
        x_upd = x_upd.at[2].set(jnp.maximum(x_upd[2], 0.01))
        x_upd = x_upd.at[3].set(jnp.maximum(x_upd[3], 0.01))
        x_upd = x_upd.at[4].set(jnp.maximum(x_upd[4], 0.01))

        return (x_upd, P_upd), x_upd

    # Run EKF for one node given (n_time, 2) observations
    def ekf_single_node(y_node):
        x0 = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
        P0 = jnp.eye(5) * 1e-2
        _, x_traj = jax.lax.scan(ekf_step, (x0, P0), y_node)
        return x_traj  # (n_time, 5)

    # vmap over nodes: y_all is (n_time, 2, nn)
    # We need (nn, n_time, 2) for vmap
    y_per_node = jnp.transpose(y_all, (2, 0, 1))  # (nn, n_time, 2)

    # JIT + vmap
    ekf_all = jax.jit(jax.vmap(ekf_single_node))(y_per_node)
    # ekf_all: (nn, n_time, 5)

    # Extract results
    x_all = np.asarray(ekf_all)  # (nn, n_time, 5)

    # Convert state to HbO/HbR
    v_all = x_all[:, :, 3]  # (nn, n_time)
    q_all = x_all[:, :, 4]
    hbo_filt = _HbT_0 * v_all - _HbR_0 * q_all - _HbO_0  # (nn, n_time)
    hbr_filt = _HbR_0 * q_all - _HbR_0
    neural_filt = x_all[:, :, 0]

    # Scale output back to input units
    inv_scale = 1.0 / scale

    return PhysiologicalResult(
        hbo=np.asarray(hbo_filt.T * inv_scale),  # (n_time, nn)
        hbr=np.asarray(hbr_filt.T * inv_scale),
        neural=np.asarray(neural_filt.T),
    )
