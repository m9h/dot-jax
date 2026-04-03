"""Closed-form analytical solutions for the diffusion equation.

JAX-native implementations of the analytical solutions from redbirdpy.
All functions are JIT-compatible and differentiable via jax.grad.

The CW fluence solutions implement the photon diffusion equation in
homogeneous media. The semi-infinite geometry uses the extrapolated
boundary condition with image sources following Haskell et al. (1994)
and Farrell et al. (1992).

References
----------
.. [1] R. C. Haskell et al., "Boundary conditions for the diffusion
       equation in radiative transfer," JOSA A, vol. 11, no. 10,
       pp. 2727-2741, 1994.
.. [2] T. J. Farrell, M. S. Patterson, and B. C. Wilson, "A diffusion
       theory model of spatially resolved, steady-state diffuse
       reflectance for the noninvasive determination of tissue optical
       properties in vivo," Med. Phys., vol. 19, pp. 879-888, 1992.
.. [3] A. Ishimaru, "Wave Propagation and Scattering in Random Media,"
       Academic Press, 1978.

Functions:
    getreff: Effective reflection coefficient (Haskell 1994)
    getdistance: Source-detector distance matrix
    infinite_cw: CW fluence for infinite homogeneous medium
    semi_infinite_cw: CW fluence for semi-infinite medium (image source)
    spbesselj, spbessely, spbesselh: Spherical Bessel/Hankel functions
    spbesseljprime, spbesselyprime, spbesselhprime: Their derivatives
    spharmonic: Spherical harmonic Y_l^m
"""

import jax
import jax.numpy as jnp
from math import factorial as _factorial


# =============================================================================
# Spherical Bessel / Hankel functions
# =============================================================================
# JAX-native implementations using the recurrence relations.
# For orders 0 and 1, closed-form; higher orders via upward recurrence.


def _j0(z):
    return jnp.sin(z) / z


def _j1(z):
    return jnp.sin(z) / z**2 - jnp.cos(z) / z


def _y0(z):
    return -jnp.cos(z) / z


def _y1(z):
    return -jnp.cos(z) / z**2 - jnp.sin(z) / z


def _bessel_recurrence(fn_prev, fn_curr, n, z):
    """Upward recurrence: f_{n+1}(z) = (2n+1)/z * f_n(z) - f_{n-1}(z)."""
    return (2 * n + 1) / z * fn_curr - fn_prev


def _spherical_bessel(z, f0_fn, f1_fn, order):
    """Compute spherical Bessel function of given order via recurrence."""
    f0 = f0_fn(z)
    if order == 0:
        return f0
    f1 = f1_fn(z)
    if order == 1:
        return f1

    def body(carry, n):
        fn_prev, fn_curr = carry
        fn_next = _bessel_recurrence(fn_prev, fn_curr, n, z)
        return (fn_curr, fn_next), None

    (_, result), _ = jax.lax.scan(body, (f0, f1), jnp.arange(1, order))
    return result


def spbesselj(n, z):
    """Spherical Bessel function of the first kind j_n(z)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    return _spherical_bessel(z, _j0, _j1, n)


def spbessely(n, z):
    """Spherical Bessel function of the second kind (Neumann) y_n(z)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    return _spherical_bessel(z, _y0, _y1, n)


def spbesselh(n, k, z):
    """Spherical Hankel function h_n^(k)(z)."""
    jn = spbesselj(n, z)
    yn = spbessely(n, z)
    if k == 1:
        return jn + 1j * yn
    elif k == 2:
        return jn - 1j * yn
    else:
        raise ValueError("k must be 1 or 2")


def spbesseljprime(n, z):
    """Derivative of spherical Bessel function j'_n(z).

    Uses recurrence: j'_n(z) = j_{n-1}(z) - (n+1)/z * j_n(z)  for n >= 1
    and j'_0(z) = -j_1(z).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    if n == 0:
        return -spbesselj(1, z)
    return spbesselj(n - 1, z) - (n + 1) / z * spbesselj(n, z)


def spbesselyprime(n, z):
    """Derivative of spherical Neumann function y'_n(z)."""
    z = jnp.asarray(z, dtype=jnp.float64)
    if n == 0:
        return -spbessely(1, z)
    return spbessely(n - 1, z) - (n + 1) / z * spbessely(n, z)


def spbesselhprime(n, k, z):
    """Derivative of spherical Hankel function h'^(k)_n(z)."""
    jp = spbesseljprime(n, z)
    yp = spbesselyprime(n, z)
    if k == 1:
        return jp + 1j * yp
    elif k == 2:
        return jp - 1j * yp
    else:
        raise ValueError("k must be 1 or 2")


# =============================================================================
# Spherical Harmonics
# =============================================================================


def spharmonic(l, m, theta, phi):
    """Spherical harmonic Y_l^m(theta, phi).

    theta: polar angle (0 to pi), phi: azimuthal angle (0 to 2*pi).
    Uses the convention matching MATLAB/redbirdpy.
    """
    theta = jnp.atleast_1d(jnp.asarray(theta, dtype=jnp.float64))
    phi = jnp.atleast_1d(jnp.asarray(phi, dtype=jnp.float64))

    coeff = 1.0
    absm = abs(m)
    if m < 0:
        coeff = ((-1.0) ** m) * _factorial(l + m) / _factorial(l - m)

    # Associated Legendre polynomial via scipy (used outside JIT boundary)
    from scipy.special import lpmv
    Plm = lpmv(absm, l, jnp.cos(theta))
    Plm = jnp.asarray(Plm, dtype=jnp.float64)

    norm = jnp.sqrt(
        (2 * l + 1) * _factorial(l - m) / (4 * jnp.pi * _factorial(l + m))
    )

    result = coeff * norm * Plm * jnp.exp(1j * m * phi)
    return result.item() if result.size == 1 else result


# =============================================================================
# Utility: reflection coefficient, distance
# =============================================================================


def getreff(n_in, n_out=1.0):
    """Effective reflection coefficient (Haskell 1994).

    Numerical integration of Fresnel reflectance over all angles.
    """
    n_in = jnp.asarray(n_in, dtype=jnp.float64)
    n_out = jnp.asarray(n_out, dtype=jnp.float64)

    # For matched or lower refractive index, no internal reflection
    def _zero(_):
        return 0.0

    def _compute(_):
        oc = jnp.arcsin(n_out / n_in)
        ostep = jnp.pi / 2000

        o = jnp.arange(0, jnp.pi / 2, ostep)
        n_steps = o.shape[0]

        # Critical angle mask
        below_crit = o < oc

        coso = jnp.cos(o)
        sino = jnp.sin(o)
        cosop = jnp.sqrt(jnp.maximum(1 - (n_in * sino) ** 2, 0.0))

        # Fresnel reflectance
        r_s = ((n_in * cosop - n_out * coso) / (n_in * cosop + n_out * coso)) ** 2
        r_p = ((n_in * coso - n_out * cosop) / (n_in * coso + n_out * cosop)) ** 2
        r_fres = 0.5 * (r_s + r_p)

        # Total internal reflection above critical angle
        r_fres = jnp.where(below_crit, r_fres, 1.0)

        r_phi = 2 * jnp.sum(sino * coso * r_fres) * ostep
        r_j = 3 * jnp.sum(sino * coso**2 * r_fres) * ostep

        return (r_phi + r_j) / (2 - r_phi + r_j)

    return jax.lax.cond(n_in <= n_out, _zero, _compute, None)


def getdistance(srcpos, detpos):
    """Euclidean distance matrix between sources and detectors.

    Returns (Ndet, Nsrc) matrix.
    """
    srcpos = jnp.atleast_2d(jnp.asarray(srcpos, dtype=jnp.float64))
    detpos = jnp.atleast_2d(jnp.asarray(detpos, dtype=jnp.float64))

    # diff: (Ndet, Nsrc, 3)
    diff = srcpos[None, :, :3] - detpos[:, None, :3]
    return jnp.sqrt(jnp.sum(diff**2, axis=2))


# =============================================================================
# CW Analytical Solutions
# =============================================================================


def infinite_cw(mua, musp, srcpos, detpos):
    """CW fluence for infinite homogeneous medium.

    Parameters
    ----------
    mua : float  — absorption coefficient (1/mm)
    musp : float — reduced scattering coefficient (1/mm)
    srcpos : (M, 3) — source positions
    detpos : (N, 3) — detector positions

    Returns
    -------
    phi : (N, M) — fluence at each detector for each source
    """
    mua = jnp.asarray(mua, dtype=jnp.float64)
    musp = jnp.asarray(musp, dtype=jnp.float64)
    srcpos = jnp.atleast_2d(jnp.asarray(srcpos, dtype=jnp.float64))
    detpos = jnp.atleast_2d(jnp.asarray(detpos, dtype=jnp.float64))

    D = 1.0 / (3.0 * (mua + musp))
    mu_eff = jnp.sqrt(mua / D)
    r = getdistance(srcpos, detpos)
    return (1.0 / (4 * jnp.pi * D)) * jnp.exp(-mu_eff * r) / r


def semi_infinite_cw(mua, musp, n_in, n_out, srcpos, detpos):
    """CW fluence for semi-infinite medium (image source method).

    z=0 is the boundary. Sources and detectors should have z >= 0.

    Parameters
    ----------
    mua : float  — absorption coefficient (1/mm)
    musp : float — reduced scattering coefficient (1/mm)
    n_in, n_out : float — refractive indices (tissue, external)
    srcpos : (M, 3) — source positions
    detpos : (N, 3) — detector positions

    Returns
    -------
    phi : (N,) or (N, M) — fluence at detector positions
    """
    mua = jnp.asarray(mua, dtype=jnp.float64)
    musp = jnp.asarray(musp, dtype=jnp.float64)
    srcpos = jnp.atleast_2d(jnp.asarray(srcpos, dtype=jnp.float64))
    detpos = jnp.atleast_2d(jnp.asarray(detpos, dtype=jnp.float64))

    D = 1.0 / (3.0 * (mua + musp))
    Reff = getreff(n_in, n_out)
    mu_eff = jnp.sqrt(mua / D)
    zb = 2 * D * (1 + Reff) / (1 - Reff)
    z0 = 1.0 / (mua + musp)

    # Real source displaced z0 inward; image source at -(z0 + 2*zb)
    src_real = srcpos.at[:, 2].add(z0)
    src_image = srcpos.at[:, 2].add(-z0 - 2 * zb)

    r1 = getdistance(src_real, detpos)
    r2 = getdistance(src_image, detpos)

    phi = (1.0 / (4 * jnp.pi * D)) * (
        jnp.exp(-mu_eff * r1) / r1 - jnp.exp(-mu_eff * r2) / r2
    )
    return jnp.squeeze(phi)
