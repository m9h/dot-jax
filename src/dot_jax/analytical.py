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
    """Spherical Bessel function j_0(z) = sin(z)/z."""
    return jnp.sin(z) / z


def _j1(z):
    """Spherical Bessel function j_1(z) = sin(z)/z^2 - cos(z)/z."""
    return jnp.sin(z) / z**2 - jnp.cos(z) / z


def _y0(z):
    """Spherical Neumann function y_0(z) = -cos(z)/z."""
    return -jnp.cos(z) / z


def _y1(z):
    """Spherical Neumann function y_1(z) = -cos(z)/z^2 - sin(z)/z."""
    return -jnp.cos(z) / z**2 - jnp.sin(z) / z


def _bessel_recurrence(fn_prev, fn_curr, n, z):
    """Apply the spherical Bessel upward recurrence relation.

    f_{n+1}(z) = (2n+1)/z * f_n(z) - f_{n-1}(z)

    Parameters
    ----------
    fn_prev : array — f_{n-1}(z).
    fn_curr : array — f_n(z).
    n : int — current order.
    z : array — argument.

    Returns
    -------
    fn_next : array — f_{n+1}(z).
    """
    return (2 * n + 1) / z * fn_curr - fn_prev


def _spherical_bessel(z, f0_fn, f1_fn, order):
    """Evaluate a spherical Bessel function at arbitrary order.

    Uses closed-form expressions for orders 0 and 1, then applies
    upward recurrence via ``jax.lax.scan`` for higher orders.

    Parameters
    ----------
    z : array — argument.
    f0_fn : callable — function returning f_0(z).
    f1_fn : callable — function returning f_1(z).
    order : int — desired order (>= 0).

    Returns
    -------
    result : array — f_{order}(z).
    """
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
    """Spherical Bessel function of the first kind j_n(z).

    Parameters
    ----------
    n : int
        Order (>= 0).
    z : array_like
        Argument.

    Returns
    -------
    jn : jnp.ndarray
        j_n(z).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    return _spherical_bessel(z, _j0, _j1, n)


def spbessely(n, z):
    """Spherical Bessel function of the second kind (Neumann) y_n(z).

    Parameters
    ----------
    n : int
        Order (>= 0).
    z : array_like
        Argument.

    Returns
    -------
    yn : jnp.ndarray
        y_n(z).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    return _spherical_bessel(z, _y0, _y1, n)


def spbesselh(n, k, z):
    """Spherical Hankel function h_n^(k)(z).

    Parameters
    ----------
    n : int
        Order (>= 0).
    k : {1, 2}
        Kind.  ``k=1`` gives h_n^(1) = j_n + i*y_n;
        ``k=2`` gives h_n^(2) = j_n - i*y_n.
    z : array_like
        Argument.

    Returns
    -------
    hn : jnp.ndarray
        h_n^(k)(z) (complex).
    """
    jn = spbesselj(n, z)
    yn = spbessely(n, z)
    if k == 1:
        return jn + 1j * yn
    elif k == 2:
        return jn - 1j * yn
    else:
        raise ValueError("k must be 1 or 2")


def spbesseljprime(n, z):
    """Derivative of the spherical Bessel function j'_n(z).

    Uses the recurrence relation:
    ``j'_n(z) = j_{n-1}(z) - (n+1)/z * j_n(z)`` for n >= 1,
    and ``j'_0(z) = -j_1(z)``.

    Parameters
    ----------
    n : int
        Order (>= 0).
    z : array_like
        Argument.

    Returns
    -------
    jprime : jnp.ndarray
        j'_n(z).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    if n == 0:
        return -spbesselj(1, z)
    return spbesselj(n - 1, z) - (n + 1) / z * spbesselj(n, z)


def spbesselyprime(n, z):
    """Derivative of the spherical Neumann function y'_n(z).

    Parameters
    ----------
    n : int
        Order (>= 0).
    z : array_like
        Argument.

    Returns
    -------
    yprime : jnp.ndarray
        y'_n(z).
    """
    z = jnp.asarray(z, dtype=jnp.float64)
    if n == 0:
        return -spbessely(1, z)
    return spbessely(n - 1, z) - (n + 1) / z * spbessely(n, z)


def spbesselhprime(n, k, z):
    """Derivative of the spherical Hankel function h'^(k)_n(z).

    Parameters
    ----------
    n : int
        Order (>= 0).
    k : {1, 2}
        Kind (see :func:`spbesselh`).
    z : array_like
        Argument.

    Returns
    -------
    hprime : jnp.ndarray
        h'^(k)_n(z) (complex).
    """
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

    Uses the real-valued convention matching MATLAB and redbirdpy.
    The associated Legendre polynomial is evaluated via scipy (outside
    JIT).

    Parameters
    ----------
    l : int
        Degree (>= 0).
    m : int
        Order (``-l <= m <= l``).
    theta : array_like
        Polar angle in radians (0 to pi).
    phi : array_like
        Azimuthal angle in radians (0 to 2*pi).

    Returns
    -------
    Ylm : complex scalar or jnp.ndarray
        Y_l^m evaluated at the given angles.
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
    """Effective reflection coefficient for the extrapolated boundary.

    Numerically integrates the Fresnel reflectance over all incidence
    angles to obtain the effective reflection coefficient R_eff used in
    the Robin boundary condition for the photon diffusion equation.
    Follows Haskell et al. (1994), Eq. 8-10.

    Parameters
    ----------
    n_in : float
        Refractive index of the scattering medium (tissue).
    n_out : float
        Refractive index of the external medium (default: air, 1.0).

    Returns
    -------
    Reff : float
        Effective internal reflection coefficient in [0, 1).
        Returns 0.0 when ``n_in <= n_out`` (no internal reflection).
    """
    n_in = jnp.asarray(n_in, dtype=jnp.float64)
    n_out = jnp.asarray(n_out, dtype=jnp.float64)

    # For matched or lower refractive index, no internal reflection
    def _zero(_):
        return 0.0

    def _compute(_):
        # Critical angle for total internal reflection (Snell's law).
        oc = jnp.arcsin(n_out / n_in)
        ostep = jnp.pi / 2000

        # Incidence angles from 0 to pi/2 (hemisphere).
        o = jnp.arange(0, jnp.pi / 2, ostep)
        n_steps = o.shape[0]

        # Mask: angles below critical angle have partial reflection;
        # above the critical angle there is total internal reflection.
        below_crit = o < oc

        coso = jnp.cos(o)
        sino = jnp.sin(o)
        # Transmitted angle cosine via Snell's law (clamped to avoid sqrt of negative).
        cosop = jnp.sqrt(jnp.maximum(1 - (n_in * sino) ** 2, 0.0))

        # Fresnel reflectance for s- and p-polarized light.
        r_s = ((n_in * cosop - n_out * coso) / (n_in * cosop + n_out * coso)) ** 2
        r_p = ((n_in * coso - n_out * cosop) / (n_in * coso + n_out * cosop)) ** 2
        r_fres = 0.5 * (r_s + r_p)

        # Above critical angle: total internal reflection (R = 1).
        r_fres = jnp.where(below_crit, r_fres, 1.0)

        # Haskell Eq. 8-9: angular integrals R_phi and R_J weighted
        # by sin(theta)*cos(theta) and sin(theta)*cos^2(theta).
        r_phi = 2 * jnp.sum(sino * coso * r_fres) * ostep
        r_j = 3 * jnp.sum(sino * coso**2 * r_fres) * ostep

        # Haskell Eq. 10: effective reflection coefficient.
        return (r_phi + r_j) / (2 - r_phi + r_j)

    return jax.lax.cond(n_in <= n_out, _zero, _compute, None)


def getdistance(srcpos, detpos):
    """Compute Euclidean distance matrix between sources and detectors.

    Parameters
    ----------
    srcpos : (n_src, 3) array_like
        Source positions.
    detpos : (n_det, 3) array_like
        Detector positions.

    Returns
    -------
    dist : (n_det, n_src) jnp.ndarray
        Pairwise Euclidean distances.
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

    # Diffusion coefficient.
    D = 1.0 / (3.0 * (mua + musp))
    Reff = getreff(n_in, n_out)
    # Effective attenuation coefficient (inverse diffusion length).
    mu_eff = jnp.sqrt(mua / D)
    # Extrapolated boundary distance: the virtual boundary is at z = -zb.
    zb = 2 * D * (1 + Reff) / (1 - Reff)
    # Transport mean free path: isotropic point source depth.
    z0 = 1.0 / (mua + musp)

    # Image source method (Haskell 1994, Farrell 1992):
    #   Real source at depth z0 below surface (positive z).
    #   Negative image source at -(z0 + 2*zb) to enforce the
    #   extrapolated boundary condition (phi = 0 at z = -zb).
    src_real = srcpos.at[:, 2].add(z0)
    src_image = srcpos.at[:, 2].add(-z0 - 2 * zb)

    r1 = getdistance(src_real, detpos)
    r2 = getdistance(src_image, detpos)

    # Green's function: real source contribution minus image source.
    phi = (1.0 / (4 * jnp.pi * D)) * (
        jnp.exp(-mu_eff * r1) / r1 - jnp.exp(-mu_eff * r2) / r2
    )
    return jnp.squeeze(phi)
