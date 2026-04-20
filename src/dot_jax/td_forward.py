"""Time-domain forward model for diffuse optical tomography.

Solves the time-dependent photon diffusion equation via FEM spatial
discretisation and Diffrax ODE integration:

    (n/c) dPhi/dt = -(K + M_a + C) Phi + b(t)

where the source b(t) is a short (~100 ps) Gaussian laser pulse.
The solution Phi(t) at detector positions gives the Distribution of
Time-of-Flight (DTOF), from which moments are extracted for TD-fNIRS
reconstruction.

The entire pipeline — from optical properties through ODE solve to
DTOF moments — is differentiable via jax.grad through Diffrax's
adjoint backpropagation.

References
----------
.. [1] S. R. Arridge et al., "A finite element approach for modeling
       photon transport in tissue," Med. Phys., 1993.
.. [2] P. Kidger, "On Neural Differential Equations," DPhil thesis,
       University of Oxford, 2022. (Diffrax framework)
.. [3] M. S. Patterson, B. Chance, and B. C. Wilson, "Time resolved
       reflectance and transmittance for the non-invasive measurement
       of tissue optical properties," Appl. Opt., 1989.

Functions:
    assemble_mass_time: Time-domain mass matrix M_t = (n/c) * M_consistent
    td_source_pulse: Normalised Gaussian laser pulse
    td_forward_cw: Full TD forward solve via Diffrax
    dtof_moments: Extract 0th, 1st, 2nd moments from DTOF
"""

from typing import NamedTuple

import diffrax
import jax
import jax.numpy as jnp

from ._types import C0
from .assembly import assemble_mass, assemble_system_cw
from .forward import assemble_rhs


class TDForwardResult(NamedTuple):
    """Result of a time-domain forward solve.

    Attributes
    ----------
    dtof : jnp.ndarray, shape (n_times, n_det)
        Distribution of Time-of-Flight at each detector.
    times : jnp.ndarray, shape (n_times,)
        Time sample points (seconds).
    phi_t : jnp.ndarray, shape (n_times, nn)
        Fluence field at all mesh nodes over time.
    """
    dtof: jnp.ndarray    # (n_times, n_det) DTOF at each detector
    times: jnp.ndarray   # (n_times,) time points
    phi_t: jnp.ndarray   # (n_times, nn) fluence field over time


def assemble_mass_time(mesh, n=1.37):
    """Assemble time-domain mass matrix.

    M_t = (n / c0) * M_consistent

    where M_consistent uses the same linear tet coefficients (1/10
    diagonal, 1/20 off-diagonal) as the absorption mass matrix.

    Parameters
    ----------
    mesh : FEMMesh
    n : float — refractive index of tissue.

    Returns
    -------
    Mt : (nn, nn) — symmetric positive semi-definite time mass matrix.
    """
    return assemble_mass(mesh, n / C0)


def td_source_pulse(t, fwhm=100e-12):
    """Normalised Gaussian laser pulse.

    Parameters
    ----------
    t : array — time points (seconds).
    fwhm : float — full width at half maximum (seconds).

    Returns
    -------
    pulse : array — normalised pulse (integrates to 1).
    """
    sigma = fwhm / 2.3548200450309493  # FWHM → sigma
    pulse = jnp.exp(-0.5 * (t / sigma) ** 2)
    # Normalise to unit integral
    norm = sigma * jnp.sqrt(2 * jnp.pi)
    return pulse / norm


def td_forward_cw(mesh, mua, musp, srcpos, detpos,
                   n_in=1.37, n_out=1.0,
                   pulse_fwhm=100e-12, t_max=5e-9, n_times=200):
    """Time-domain forward solve via Diffrax.

    Solves the time-dependent diffusion equation:
        M_t dPhi/dt = -A Phi + b * pulse(t)

    using an adaptive implicit ODE solver.

    Parameters
    ----------
    mesh : FEMMesh
    mua : float — absorption coefficient (1/mm).
    musp : float — reduced scattering coefficient (1/mm).
    srcpos : (n_src, 3) — source positions (first source used).
    detpos : (n_det, 3) — detector positions.
    n_in : float — tissue refractive index.
    n_out : float — external refractive index.
    pulse_fwhm : float — laser pulse FWHM (seconds).
    t_max : float — maximum simulation time (seconds).
    n_times : int — number of output timepoints.

    Returns
    -------
    TDForwardResult(dtof, times, phi_t)
        dtof : (n_times, n_det) — DTOF at each detector.
        times : (n_times,) — time points.
        phi_t : (n_times, nn) — fluence field over time.
    """
    # Assemble spatial operators
    A = assemble_system_cw(mesh, mua, musp, n_in, n_out)
    Mt = assemble_mass_time(mesh, n_in)

    # Source and detector vectors
    rhs_src = assemble_rhs(mesh, srcpos)
    rhs_det = assemble_rhs(mesh, detpos)
    b = rhs_src[:, 0]  # first source

    # Pre-multiply by M_t^{-1} to convert the ODE to explicit form:
    #   dPhi/dt = -M_t^{-1} A Phi + M_t^{-1} b * pulse(t)
    # This avoids solving a linear system at every RK step.
    Mt_inv_A = jnp.linalg.solve(Mt, A)
    Mt_inv_b = jnp.linalg.solve(Mt, b)

    # Gaussian pulse sigma (FWHM = 2*sqrt(2*ln2)*sigma).
    sigma = pulse_fwhm / 2.3548200450309493

    def vector_field(t, y, args):
        # Normalised Gaussian laser pulse at time t.
        pulse = jnp.exp(-0.5 * (t / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
        # ODE right-hand side: diffusion/absorption decay + source injection.
        return -Mt_inv_A @ y + Mt_inv_b * pulse

    # Uniformly spaced output time points.
    ts = jnp.linspace(0, t_max, n_times)

    # Solve with Diffrax Tsit5 (5th-order Runge-Kutta with adaptive stepping).
    # The initial step size is 1/10 of the pulse FWHM to resolve the source.
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=t_max,
        dt0=pulse_fwhm / 10,
        y0=jnp.zeros(mesh.nn),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=16384,
    )

    phi_t = sol.ys  # (n_times, nn)

    # Project the full-field fluence onto detector positions to get the
    # Distribution of Time-of-Flight (DTOF) at each detector.
    dtof = phi_t @ rhs_det  # (n_times, n_det)

    return TDForwardResult(dtof=dtof, times=ts, phi_t=phi_t)


def dtof_moments(dtof, times):
    """Extract moments from the Distribution of Time-of-Flight.

    Parameters
    ----------
    dtof : (n_times, n_det) — DTOF curves.
    times : (n_times,) — time points.

    Returns
    -------
    m0 : (n_det,) — 0th moment (total photon count, ~CW intensity).
    m1 : (n_det,) — 1st moment (mean time-of-flight).
    m2 : (n_det,) — 2nd moment (variance of time-of-flight).
    """
    # Time step widths for trapezoidal-like integration.
    dt = jnp.diff(times, prepend=times[0])

    # 0th moment: total photon count (proportional to CW intensity).
    #   m0 = integral DTOF(t) dt
    m0 = jnp.sum(dtof * dt[:, None], axis=0)

    # 1st moment: mean time-of-flight (centroid of the DTOF).
    #   m1 = integral t * DTOF(t) dt  /  m0
    m1 = jnp.sum(dtof * times[:, None] * dt[:, None], axis=0) / jnp.maximum(m0, 1e-30)

    # 2nd central moment: variance of the time-of-flight distribution.
    #   m2 = integral (t - m1)^2 * DTOF(t) dt  /  m0
    m2 = jnp.sum(dtof * (times[:, None] - m1[None, :]) ** 2 * dt[:, None], axis=0) / jnp.maximum(m0, 1e-30)

    return m0, m1, m2


def compute_moment_jacobian(mesh, mua, musp, srcpos, detpos,
                            n_in=1.37, n_out=1.0,
                            pulse_fwhm=100e-12, t_max=5e-9, n_times=200):
    """Compute Jacobian of DTOF moments w.r.t. nodal absorption.

    Uses JAX autodiff through the entire TD pipeline:
        per-node mua → assembly → ODE solve → DTOF → moments

    The Jacobian rows are [d(m0)/d(mua), d(m1)/d(mua), d(m2)/d(mua)]
    stacked for each detector.

    Parameters
    ----------
    mesh : FEMMesh
    mua : float — baseline absorption coefficient.
    musp : float — baseline reduced scattering coefficient.
    srcpos : (n_src, 3) — source positions (first source used).
    detpos : (n_det, 3) — detector positions.
    n_in, n_out : float — refractive indices.
    pulse_fwhm : float — laser pulse FWHM (seconds).
    t_max : float — simulation time (seconds).
    n_times : int — output timepoints.

    Returns
    -------
    J : (3 * n_det, nn) — Jacobian matrix.
        Rows ordered: [m0_det0, ..., m0_detN, m1_det0, ..., m2_detN].
    """
    nn = mesh.nn
    n_det = detpos.shape[0]

    # Assemble pieces that don't depend on mua (geometry-only)
    rhs_src = assemble_rhs(mesh, srcpos)
    rhs_det = assemble_rhs(mesh, detpos)
    b = rhs_src[:, 0]  # first source
    Mt = assemble_mass_time(mesh, n_in)
    sigma = pulse_fwhm / 2.3548200450309493
    ts = jnp.linspace(0, t_max, n_times)

    def moments_from_nodal_mua(mua_nodal):
        """Full TD pipeline: nodal mua → moments (differentiable)."""
        # Convert per-node → per-element (mean of 4 vertices)
        mua_elem = jnp.mean(mua_nodal[mesh.elem], axis=1)
        A = assemble_system_cw(mesh, mua_elem, musp, n_in, n_out)
        Mt_inv_A = jnp.linalg.solve(Mt, A)
        Mt_inv_b = jnp.linalg.solve(Mt, b)

        def vector_field(t, y, args):
            pulse = jnp.exp(-0.5 * (t / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
            return -Mt_inv_A @ y + Mt_inv_b * pulse

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=0.0, t1=t_max,
            dt0=pulse_fwhm / 10,
            y0=jnp.zeros(nn),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=16384,
        )
        phi_t = sol.ys  # (n_times, nn)
        dtof = phi_t @ rhs_det  # (n_times, n_det)
        m0, m1, m2 = dtof_moments(dtof, ts)
        return jnp.concatenate([m0, m1, m2])  # (3 * n_det,)

    # Compute Jacobian via reverse-mode autodiff
    mua_bg = jnp.full(nn, mua)
    J = jax.jacrev(moments_from_nodal_mua)(mua_bg)  # (3*n_det, nn)

    return J


def reconstruct_td(mesh, delta_moments, srcpos, detpos,
                   mua0, musp, n_in=1.37, n_out=1.0,
                   pulse_fwhm=100e-12, t_max=5e-9, n_times=200,
                   reg_param=1e-4):
    """Reconstruct nodal absorption perturbation from TD moment changes.

    Linearised reconstruction using the moment Jacobian:
        delta_mua = (J^T J + lambda I)^{-1} J^T delta_moments

    Parameters
    ----------
    mesh : FEMMesh
    delta_moments : (3 * n_det,) — moment perturbations [dm0, dm1, dm2].
    srcpos, detpos : source/detector positions.
    mua0 : float — background absorption for Jacobian linearisation.
    musp : float — reduced scattering (fixed).
    n_in, n_out : float — refractive indices.
    pulse_fwhm, t_max, n_times : TD parameters.
    reg_param : float — Tikhonov regularisation.

    Returns
    -------
    delta_mua : (nn,) — reconstructed nodal absorption perturbation.
    """
    J = compute_moment_jacobian(
        mesh, mua0, musp, srcpos, detpos, n_in, n_out,
        pulse_fwhm, t_max, n_times,
    )

    nn = J.shape[1]
    JtJ = J.T @ J
    Jtd = J.T @ delta_moments
    delta_mua = jnp.linalg.solve(JtJ + reg_param * jnp.eye(nn), Jtd)

    return delta_mua
