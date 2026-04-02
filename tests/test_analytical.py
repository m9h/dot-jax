"""Tests for analytical diffusion solutions — RED-GREEN TDD.

Cross-validates against redbirdpy and known mathematical identities.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from dot_jax.analytical import (
    getreff,
    getdistance,
    infinite_cw,
    semi_infinite_cw,
    spbesselj,
    spbessely,
    spbesselh,
    spbesseljprime,
    spbesselyprime,
    spbesselhprime,
    spharmonic,
)


# ---------------------------------------------------------------------------
# Spherical Bessel functions
# ---------------------------------------------------------------------------

class TestSphericalBesselJ:
    def test_order0_known_value(self):
        """j_0(z) = sin(z)/z."""
        z = 1.0
        expected = jnp.sin(z) / z
        npt.assert_allclose(spbesselj(0, z), expected, atol=1e-10)

    def test_order1_known_value(self):
        """j_1(z) = sin(z)/z^2 - cos(z)/z."""
        z = 2.0
        expected = jnp.sin(z) / z**2 - jnp.cos(z) / z
        npt.assert_allclose(spbesselj(1, z), expected, atol=1e-10)

    def test_cross_validate_scipy(self):
        from scipy.special import spherical_jn
        for n in range(6):
            for z in [0.5, 1.0, 2.0, 5.0, 10.0]:
                expected = spherical_jn(n, z)
                npt.assert_allclose(
                    float(spbesselj(n, z)), float(expected), rtol=1e-6,
                    err_msg=f"n={n}, z={z}",
                )

    def test_jit(self):
        f = jax.jit(lambda z: spbesselj(0, z))
        npt.assert_allclose(f(1.0), spbesselj(0, 1.0), rtol=1e-12)

    def test_vmap_over_z(self):
        z = jnp.linspace(0.1, 10.0, 50)
        result = jax.vmap(lambda zi: spbesselj(0, zi))(z)
        assert result.shape == (50,)
        assert jnp.all(jnp.isfinite(result))


class TestSphericalBesselY:
    def test_order0_known_value(self):
        """y_0(z) = -cos(z)/z."""
        z = 1.0
        expected = -jnp.cos(z) / z
        npt.assert_allclose(spbessely(0, z), expected, atol=1e-10)


class TestSphericalHankel:
    def test_kind1_identity(self):
        """h_n^(1)(z) = j_n(z) + i*y_n(z)."""
        z = 1.5
        for n in range(4):
            expected = spbesselj(n, z) + 1j * spbessely(n, z)
            npt.assert_allclose(spbesselh(n, 1, z), expected, atol=1e-10)

    def test_kind2_identity(self):
        """h_n^(2)(z) = j_n(z) - i*y_n(z)."""
        z = 1.5
        for n in range(4):
            expected = spbesselj(n, z) - 1j * spbessely(n, z)
            npt.assert_allclose(spbesselh(n, 2, z), expected, atol=1e-10)


class TestSphericalBesselDerivatives:
    def test_jprime_order0_recurrence(self):
        """j'_0(z) = -j_1(z)."""
        z = 1.0
        npt.assert_allclose(
            spbesseljprime(0, z), -spbesselj(1, z), atol=1e-10,
        )

    def test_yprime_cross_validate_scipy(self):
        from scipy.special import spherical_yn
        z = 2.0
        expected = spherical_yn(0, z, derivative=True)
        npt.assert_allclose(float(spbesselyprime(0, z)), float(expected), rtol=1e-12)

    def test_hprime_kind1(self):
        z = 1.0
        expected = spbesseljprime(0, z) + 1j * spbesselyprime(0, z)
        npt.assert_allclose(spbesselhprime(0, 1, z), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Spherical harmonics
# ---------------------------------------------------------------------------

class TestSphericalHarmonics:
    def test_Y00_constant(self):
        """Y_0^0 = 1/sqrt(4*pi)."""
        expected = 1.0 / jnp.sqrt(4 * jnp.pi)
        result = spharmonic(0, 0, 0.5, 0.0)
        npt.assert_allclose(jnp.real(result), expected, atol=1e-10)

    def test_Y10_cosine(self):
        """Y_1^0(theta, 0) = sqrt(3/(4*pi)) * cos(theta)."""
        theta = jnp.pi / 4
        expected = jnp.sqrt(3.0 / (4.0 * jnp.pi)) * jnp.cos(theta)
        result = spharmonic(1, 0, theta, 0.0)
        npt.assert_allclose(jnp.real(result), expected, atol=1e-10)

    def test_cross_validate_redbirdpy(self):
        from redbirdpy.analytical import spharmonic as spharmonic_rb
        for l in range(4):
            for m in range(-l, l + 1):
                theta, phi = 0.7, 1.2
                expected = spharmonic_rb(l, m, theta, phi)
                result = spharmonic(l, m, theta, phi)
                npt.assert_allclose(
                    complex(result), complex(expected), rtol=1e-10,
                    err_msg=f"l={l}, m={m}",
                )


# ---------------------------------------------------------------------------
# Utility: getreff, getdistance
# ---------------------------------------------------------------------------

class TestGetreff:
    def test_matched_indices_zero(self):
        assert getreff(1.0, 1.0) == 0.0

    def test_tissue_air_known_value(self):
        from redbirdpy.utility import getreff as getreff_rb
        expected = getreff_rb(1.37, 1.0)
        result = getreff(1.37, 1.0)
        npt.assert_allclose(result, expected, rtol=1e-4)

    def test_cross_validate_numpy(self):
        from redbirdpy.utility import getreff as getreff_rb
        for n_in in [1.33, 1.37, 1.4, 1.5]:
            expected = getreff_rb(n_in, 1.0)
            npt.assert_allclose(
                float(getreff(n_in, 1.0)), float(expected), rtol=1e-6,
                err_msg=f"n_in={n_in}",
            )

    def test_jit(self):
        f = jax.jit(getreff)
        npt.assert_allclose(f(1.37, 1.0), getreff(1.37, 1.0), rtol=1e-12)


class TestGetdistance:
    def test_known_values(self):
        src = jnp.array([[0.0, 0.0, 0.0]])
        det = jnp.array([[3.0, 4.0, 0.0]])
        d = getdistance(src, det)
        npt.assert_allclose(d, 5.0, atol=1e-10)

    def test_multiple_detectors(self):
        src = jnp.array([[0.0, 0.0, 0.0]])
        det = jnp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        d = getdistance(src, det)
        assert d.shape == (2, 1)
        npt.assert_allclose(d[:, 0], [10.0, 20.0], atol=1e-10)


# ---------------------------------------------------------------------------
# CW solutions
# ---------------------------------------------------------------------------

class TestInfiniteCW:
    def test_positive_fluence(self, tissue_props, simple_positions):
        phi = infinite_cw(
            tissue_props["mua"], tissue_props["musp"],
            simple_positions["srcpos"], simple_positions["detpos"],
        )
        assert jnp.all(phi > 0)

    def test_radial_decay(self, tissue_props):
        src = jnp.array([[0.0, 0.0, 0.0]])
        det = jnp.array([
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ])
        phi = infinite_cw(tissue_props["mua"], tissue_props["musp"], src, det)
        assert phi[0] > phi[1] > phi[2]

    def test_spherical_symmetry(self, tissue_props):
        src = jnp.array([[0.0, 0.0, 0.0]])
        det = jnp.array([
            [10.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ])
        phi = infinite_cw(tissue_props["mua"], tissue_props["musp"], src, det)
        npt.assert_allclose(phi[0], phi[1], rtol=1e-10)
        npt.assert_allclose(phi[0], phi[2], rtol=1e-10)

    def test_cross_validate_numpy(self, tissue_props, simple_positions):
        from redbirdpy.analytical import infinite_cw as infinite_cw_rb
        expected = infinite_cw_rb(
            tissue_props["mua"], tissue_props["musp"],
            np.array(simple_positions["srcpos"]),
            np.array(simple_positions["detpos"]),
        )
        result = infinite_cw(
            tissue_props["mua"], tissue_props["musp"],
            simple_positions["srcpos"], simple_positions["detpos"],
        )
        npt.assert_allclose(np.array(result), expected, rtol=1e-12)

    def test_grad_wrt_mua(self, tissue_props, simple_positions):
        def phi_sum(mua):
            return jnp.sum(infinite_cw(
                mua, tissue_props["musp"],
                simple_positions["srcpos"], simple_positions["detpos"],
            ))
        grad = jax.grad(phi_sum)(tissue_props["mua"])
        assert jnp.isfinite(grad)
        assert grad < 0  # more absorption → less fluence

    def test_grad_wrt_musp(self, tissue_props, simple_positions):
        def phi_sum(musp):
            return jnp.sum(infinite_cw(
                tissue_props["mua"], musp,
                simple_positions["srcpos"], simple_positions["detpos"],
            ))
        grad = jax.grad(phi_sum)(tissue_props["musp"])
        assert jnp.isfinite(grad)
        # Sign depends on distance: near detectors get more signal from
        # increased scattering (photons stay near surface), so just check finite

    def test_jit(self, tissue_props, simple_positions):
        f = jax.jit(lambda mua, musp: infinite_cw(
            mua, musp,
            simple_positions["srcpos"], simple_positions["detpos"],
        ))
        eager = infinite_cw(
            tissue_props["mua"], tissue_props["musp"],
            simple_positions["srcpos"], simple_positions["detpos"],
        )
        jitted = f(tissue_props["mua"], tissue_props["musp"])
        npt.assert_allclose(jitted, eager, rtol=1e-12)

    def test_vmap_over_detectors(self, tissue_props):
        src = jnp.array([[0.0, 0.0, 0.0]])
        dets = jax.random.uniform(jax.random.PRNGKey(0), (100, 3), minval=1.0, maxval=50.0)

        def single_det(det):
            return infinite_cw(
                tissue_props["mua"], tissue_props["musp"],
                src, det[None, :],
            )[0]

        result = jax.vmap(single_det)(dets)
        assert result.shape == (100,) or result.shape == (100, 1)
        assert jnp.all(jnp.isfinite(result))


class TestSemiInfiniteCW:
    def test_less_than_infinite(self, tissue_props):
        src = jnp.array([[30.0, 30.0, 0.0]])
        det = jnp.array([[30.0, 40.0, 0.0]])
        phi_inf = infinite_cw(tissue_props["mua"], tissue_props["musp"], src, det)
        phi_semi = semi_infinite_cw(
            tissue_props["mua"], tissue_props["musp"],
            tissue_props["n_in"], tissue_props["n_out"],
            src, det,
        )
        assert phi_semi < phi_inf

    def test_cross_validate_numpy(self, tissue_props):
        from redbirdpy.analytical import semi_infinite_cw as semi_cw_rb
        src = jnp.array([[30.0, 30.0, 0.0]])
        det = jnp.array([[30.0, 40.0, 0.0], [40.0, 30.0, 0.0]])
        expected = semi_cw_rb(
            tissue_props["mua"], tissue_props["musp"],
            tissue_props["n_in"], tissue_props["n_out"],
            np.array(src), np.array(det),
        )
        result = semi_infinite_cw(
            tissue_props["mua"], tissue_props["musp"],
            tissue_props["n_in"], tissue_props["n_out"],
            src, det,
        )
        npt.assert_allclose(np.array(result), expected, rtol=1e-10)

    def test_grad_wrt_mua(self, tissue_props):
        src = jnp.array([[30.0, 30.0, 0.0]])
        det = jnp.array([[30.0, 40.0, 0.0]])

        def phi_val(mua):
            return jnp.sum(semi_infinite_cw(
                mua, tissue_props["musp"],
                tissue_props["n_in"], tissue_props["n_out"],
                src, det,
            ))
        grad = jax.grad(phi_val)(tissue_props["mua"])
        assert jnp.isfinite(grad)
        assert grad < 0

    def test_jit(self, tissue_props):
        src = jnp.array([[30.0, 30.0, 0.0]])
        det = jnp.array([[30.0, 40.0, 0.0]])
        f = jax.jit(lambda mua: semi_infinite_cw(
            mua, tissue_props["musp"],
            tissue_props["n_in"], tissue_props["n_out"],
            src, det,
        ))
        eager = semi_infinite_cw(
            tissue_props["mua"], tissue_props["musp"],
            tissue_props["n_in"], tissue_props["n_out"],
            src, det,
        )
        npt.assert_allclose(f(tissue_props["mua"]), eager, rtol=1e-12)
