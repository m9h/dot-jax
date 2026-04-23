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


def _solve_td_multi_source(mesh, mua, musp, b_all, n_in, n_out,
                           pulse_fwhm, t_max, n_times):
    """Solve the TD diffusion ODE for many sources, sharing A assembly.

    The expensive pieces — FEM system assembly and the Mt^{-1} A solve —
    do not depend on the source, so they are computed once and reused
    across all sources via vmap.

    Parameters
    ----------
    mesh : FEMMesh
    mua : float or (ne,) — absorption.
    musp : float or (ne,) — reduced scattering.
    b_all : (n_src, nn) — stacked RHS vectors, one per source.
    n_in, n_out : float — refractive indices.
    pulse_fwhm, t_max, n_times : TD config.

    Returns
    -------
    phi_t_all : (n_src, n_times, nn) — fluence field per source and time.
    ts : (n_times,) — output time samples.
    """
    A = assemble_system_cw(mesh, mua, musp, n_in, n_out)
    Mt = assemble_mass_time(mesh, n_in)
    Mt_inv_A = jnp.linalg.solve(Mt, A)                   # (nn, nn), shared.
    Mt_inv_b_all = jnp.linalg.solve(Mt, b_all.T).T       # (n_src, nn).

    sigma = pulse_fwhm / 2.3548200450309493
    ts = jnp.linspace(0, t_max, n_times)

    def solve_one(Mt_inv_b):
        def vector_field(t, y, args):
            pulse = jnp.exp(-0.5 * (t / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
            return -Mt_inv_A @ y + Mt_inv_b * pulse
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=0.0, t1=t_max,
            dt0=pulse_fwhm / 10,
            y0=jnp.zeros(mesh.nn),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=16384,
        )
        return sol.ys

    phi_t_all = jax.vmap(solve_one)(Mt_inv_b_all)
    return phi_t_all, ts


def _solve_td_ode(mesh, mua, musp, b, n_in, n_out,
                  pulse_fwhm, t_max, n_times):
    """Solve the TD diffusion ODE for a single source injection.

    Solves  M_t dPhi/dt = -A Phi + b * pulse(t)  on (0, t_max) by
    pre-multiplying with M_t^{-1} and integrating the resulting
    explicit ODE with Diffrax's Tsit5 adaptive RK method.

    Parameters
    ----------
    mesh : FEMMesh
    mua : float or (ne,) — absorption (per-element or scalar).
    musp : float or (ne,) — reduced scattering (per-element or scalar).
    b : (nn,) — RHS vector for this source.
    n_in, n_out : float — refractive indices.
    pulse_fwhm, t_max, n_times : TD config.

    Returns
    -------
    phi_t : (n_times, nn) — fluence field at each time sample.
    ts : (n_times,) — output time samples.
    """
    A = assemble_system_cw(mesh, mua, musp, n_in, n_out)
    Mt = assemble_mass_time(mesh, n_in)

    # Pre-multiply by M_t^{-1} to get explicit form dPhi/dt = -M_t^{-1}A Phi + M_t^{-1}b p(t).
    # Avoids solving a linear system at every RK step.
    Mt_inv_A = jnp.linalg.solve(Mt, A)
    Mt_inv_b = jnp.linalg.solve(Mt, b)

    sigma = pulse_fwhm / 2.3548200450309493

    def vector_field(t, y, args):
        pulse = jnp.exp(-0.5 * (t / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
        return -Mt_inv_A @ y + Mt_inv_b * pulse

    ts = jnp.linspace(0, t_max, n_times)
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
    return sol.ys, ts


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
    mua : float or (ne,) — absorption coefficient (1/mm).
    musp : float or (ne,) — reduced scattering coefficient (1/mm).
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
    rhs_src = assemble_rhs(mesh, srcpos)
    rhs_det = assemble_rhs(mesh, detpos)
    b = rhs_src[:, 0]  # first source

    phi_t, ts = _solve_td_ode(
        mesh, mua, musp, b, n_in, n_out,
        pulse_fwhm, t_max, n_times,
    )

    # Project fluence onto detector positions.
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
                            pulse_fwhm=100e-12, t_max=5e-9, n_times=200,
                            *, channel_pairs=None):
    """Compute Jacobian of DTOF moments w.r.t. nodal absorption.

    Uses JAX autodiff through the entire TD pipeline:
        per-node mua → assembly → ODE solve → DTOF → moments

    Supports multiple sources, sharing the FEM assembly and Mt^{-1} A
    solve across sources via vmap. The Jacobian rows are
    ``[d(m0)/d(mua); d(m1)/d(mua); d(m2)/d(mua)]`` concatenated in that
    moment order, each spanning all channels.

    Parameters
    ----------
    mesh : FEMMesh
    mua : float — baseline absorption coefficient.
    musp : float — baseline reduced scattering coefficient.
    srcpos : (n_src, 3) — source positions.
    detpos : (n_det, 3) — detector positions.
    channel_pairs : (n_ch, 2) int array, optional
        Explicit (src_idx, det_idx) pairs to include, matching an SNIRF
        measurement list. When omitted, the full Cartesian product of
        sources and detectors is used, yielding ``n_src * n_det`` channels.
    n_in, n_out : float — refractive indices.
    pulse_fwhm : float — laser pulse FWHM (seconds).
    t_max : float — simulation time (seconds).
    n_times : int — output timepoints.

    Returns
    -------
    J : (3 * n_ch, nn) — Jacobian matrix.
        Rows ordered ``[m0 (all channels); m1 (all channels); m2 (all channels)]``,
        where channel indexing follows ``channel_pairs`` when given, or
        row-major enumeration of (src, det) pairs otherwise.
    """
    nn = mesh.nn
    n_src = srcpos.shape[0]
    n_det = detpos.shape[0]

    # Geometry-only pieces: invariant under mua.
    rhs_src = assemble_rhs(mesh, srcpos)    # (nn, n_src)
    rhs_det = assemble_rhs(mesh, detpos)    # (nn, n_det)
    b_all = rhs_src.T                       # (n_src, nn)

    if channel_pairs is not None:
        channel_pairs = jnp.asarray(channel_pairs, dtype=jnp.int32)
        src_sel = channel_pairs[:, 0]
        det_sel = channel_pairs[:, 1]

    def moments_from_nodal_mua(mua_nodal):
        """Full TD pipeline: nodal mua → moments (differentiable)."""
        mua_elem = jnp.mean(mua_nodal[mesh.elem], axis=1)
        phi_t_all, ts = _solve_td_multi_source(
            mesh, mua_elem, musp, b_all, n_in, n_out,
            pulse_fwhm, t_max, n_times,
        )  # (n_src, n_times, nn)
        # Project each source's field onto all detectors: dtof[s, t, d].
        dtof_all = jnp.einsum('stn,nd->std', phi_t_all, rhs_det)

        if channel_pairs is None:
            # All (src, det) pairs, row-major in src then det:
            # dtof_ch[:, s*n_det + d] = dtof_all[s, :, d].
            dtof_ch = dtof_all.transpose(1, 0, 2).reshape(n_times, n_src * n_det)
        else:
            # Gather the requested (src, det) pairs.
            dtof_ch = dtof_all[src_sel, :, det_sel].T  # (n_times, n_ch)

        m0, m1, m2 = dtof_moments(dtof_ch, ts)
        return jnp.concatenate([m0, m1, m2])

    mua_bg = jnp.full(nn, mua)
    return jax.jacrev(moments_from_nodal_mua)(mua_bg)


def reconstruct_td(mesh, delta_moments, srcpos, detpos,
                   mua0, musp, n_in=1.37, n_out=1.0,
                   pulse_fwhm=100e-12, t_max=5e-9, n_times=200,
                   reg_param=1e-4, *, channel_pairs=None):
    """Reconstruct nodal absorption perturbation from TD moment changes.

    Linearised reconstruction using the moment Jacobian:
        delta_mua = (J^T J + lambda I)^{-1} J^T delta_moments

    Primal Tikhonov form. For underdetermined problems where ``nn >>
    n_meas`` (atlas-scale meshes), prefer :func:`reconstruct_td_dual`.

    Parameters
    ----------
    mesh : FEMMesh
    delta_moments : (3 * n_ch,) — moment perturbations [dm0, dm1, dm2].
    srcpos, detpos : source/detector positions.
    mua0 : float — background absorption for Jacobian linearisation.
    musp : float — reduced scattering (fixed).
    n_in, n_out : float — refractive indices.
    pulse_fwhm, t_max, n_times : TD parameters.
    reg_param : float — Tikhonov regularisation.
    channel_pairs : (n_ch, 2) int, optional
        Explicit (src_idx, det_idx) channel list, passed through to
        :func:`compute_moment_jacobian`.

    Returns
    -------
    delta_mua : (nn,) — reconstructed nodal absorption perturbation.
    """
    J = compute_moment_jacobian(
        mesh, mua0, musp, srcpos, detpos, n_in, n_out,
        pulse_fwhm, t_max, n_times,
        channel_pairs=channel_pairs,
    )

    nn = J.shape[1]
    JtJ = J.T @ J
    Jtd = J.T @ delta_moments
    delta_mua = jnp.linalg.solve(JtJ + reg_param * jnp.eye(nn), Jtd)

    return delta_mua


def reconstruct_td_dual(mesh, delta_moments, srcpos, detpos,
                        mua0, musp, n_in=1.37, n_out=1.0,
                        pulse_fwhm=100e-12, t_max=5e-9, n_times=200,
                        reg_param=0.01, *, channel_pairs=None):
    """Dual-formulation TD reconstruction for underdetermined problems.

    Computes the moment Jacobian via autodiff, then solves
        delta_mua = J^T (J J^T + lambda * sqrt(||J J^T||) * I)^{-1} delta_moments
    using :func:`dot_jax.recon.solve_dual`. Efficient when the number of
    mesh nodes greatly exceeds the number of channels — the matrix
    inverted is ``(n_meas, n_meas)`` rather than ``(nn, nn)``. This is
    the formulation used by the Kernel/Holoscan BCI pipeline at the
    optical-property layer; here we apply it to TD moments.

    Parameters
    ----------
    mesh : FEMMesh
    delta_moments : (3 * n_ch,) — moment perturbations [dm0, dm1, dm2].
    srcpos, detpos : source/detector positions.
    mua0, musp, n_in, n_out, pulse_fwhm, t_max, n_times : forward
        configuration (see :func:`compute_moment_jacobian`).
    reg_param : float — relative Tikhonov regularisation (scaled by
        ``sqrt(||J J^T||)`` inside ``solve_dual``).
    channel_pairs : (n_ch, 2) int, optional
        Explicit (src_idx, det_idx) channel list.

    Returns
    -------
    delta_mua : (nn,) — reconstructed nodal absorption perturbation.
    """
    from .recon import solve_dual

    J = compute_moment_jacobian(
        mesh, mua0, musp, srcpos, detpos, n_in, n_out,
        pulse_fwhm, t_max, n_times,
        channel_pairs=channel_pairs,
    )
    return solve_dual(J, delta_moments, reg=reg_param)
