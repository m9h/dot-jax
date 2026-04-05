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
    """Result of a time-domain forward solve."""
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

    # Precompute M_t^{-1} A and M_t^{-1} b for efficiency
    Mt_inv_A = jnp.linalg.solve(Mt, A)
    Mt_inv_b = jnp.linalg.solve(Mt, b)

    sigma = pulse_fwhm / 2.3548200450309493

    def vector_field(t, y, args):
        pulse = jnp.exp(-0.5 * (t / sigma) ** 2) / (sigma * jnp.sqrt(2 * jnp.pi))
        return -Mt_inv_A @ y + Mt_inv_b * pulse

    # Save at specified timepoints
    ts = jnp.linspace(0, t_max, n_times)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=0.0,
        t1=t_max,
        dt0=pulse_fwhm / 10,  # initial step: 1/10 of pulse width
        y0=jnp.zeros(mesh.nn),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=16384,
    )

    phi_t = sol.ys  # (n_times, nn)

    # DTOF: project fluence onto detectors
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
    dt = jnp.diff(times, prepend=times[0])

    # 0th moment: integral of DTOF
    m0 = jnp.sum(dtof * dt[:, None], axis=0)

    # 1st moment: mean time-of-flight
    m1 = jnp.sum(dtof * times[:, None] * dt[:, None], axis=0) / jnp.maximum(m0, 1e-30)

    # 2nd moment: variance
    m2 = jnp.sum(dtof * (times[:, None] - m1[None, :]) ** 2 * dt[:, None], axis=0) / jnp.maximum(m0, 1e-30)

    return m0, m1, m2
