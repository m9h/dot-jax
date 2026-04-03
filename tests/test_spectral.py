"""Tests for multi-spectral chromophore decomposition — RED-GREEN TDD.

Validates spectral forward model and Jacobian computation across wavelengths.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from dot_jax.mesh import FEMMesh
from dot_jax.property import extinction, mua_from_concentrations, musp_from_scattering
from dot_jax.spectral import (
    spectral_forward_cw,
    compute_jacobian_mua,
)


@pytest.fixture
def box_mesh():
    """20x20x20 mm box mesh for spectral testing."""
    node = jnp.array([
        [0.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [20.0, 20.0, 0.0],
        [0.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [0.0, 20.0, 20.0],
        [20.0, 20.0, 20.0],
    ], dtype=jnp.float64)
    elem = jnp.array([
        [0, 1, 2, 4],
        [1, 2, 3, 7],
        [1, 2, 4, 7],
        [1, 4, 5, 7],
        [2, 4, 6, 7],
    ], dtype=jnp.int32)
    return FEMMesh.create(node, elem)


@pytest.fixture
def spectral_setup():
    """Standard two-wavelength NIR setup."""
    wavelengths = jnp.array([690.0, 830.0])
    E = extinction([690, 830], ["hbo", "hbr"])
    return {"wavelengths": wavelengths, "E": E}


# ---------------------------------------------------------------------------
# Spectral forward model
# ---------------------------------------------------------------------------

class TestSpectralForwardCW:
    def test_shape(self, box_mesh, spectral_setup):
        """Should return detval per wavelength."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        mua_wv = jnp.array([0.01, 0.012])  # mua at each wavelength
        musp_wv = jnp.array([1.0, 0.9])     # musp at each wavelength

        detvals = spectral_forward_cw(
            box_mesh, mua_wv, musp_wv, srcpos, detpos,
        )
        assert detvals.shape == (2, 1, 1)  # (n_wv, n_det, n_src)

    def test_positive_detvals(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        mua_wv = jnp.array([0.01, 0.012])
        musp_wv = jnp.array([1.0, 0.9])

        detvals = spectral_forward_cw(box_mesh, mua_wv, musp_wv, srcpos, detpos)
        assert jnp.all(detvals > 0)

    def test_different_wavelengths_differ(self, box_mesh):
        """Different optical properties should give different detector values."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        mua_wv = jnp.array([0.01, 0.05])  # very different absorption
        musp_wv = jnp.array([1.0, 0.5])

        detvals = spectral_forward_cw(box_mesh, mua_wv, musp_wv, srcpos, detpos)
        assert detvals[0, 0, 0] != detvals[1, 0, 0]

    def test_multi_source_detector(self, box_mesh):
        srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
        detpos = jnp.array([[10.0, 10.0, 10.0], [15.0, 5.0, 5.0]])
        mua_wv = jnp.array([0.01, 0.012])
        musp_wv = jnp.array([1.0, 0.9])

        detvals = spectral_forward_cw(box_mesh, mua_wv, musp_wv, srcpos, detpos)
        assert detvals.shape == (2, 2, 2)  # (n_wv, n_det, n_src)

    def test_finite(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        mua_wv = jnp.array([0.01, 0.012])
        musp_wv = jnp.array([1.0, 0.9])

        detvals = spectral_forward_cw(box_mesh, mua_wv, musp_wv, srcpos, detpos)
        assert jnp.all(jnp.isfinite(detvals))

    def test_grad_wrt_mua(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        musp_wv = jnp.array([1.0, 0.9])

        def total_signal(mua_wv):
            return jnp.sum(spectral_forward_cw(
                box_mesh, mua_wv, musp_wv, srcpos, detpos,
            ))

        g = jax.grad(total_signal)(jnp.array([0.01, 0.012]))
        assert jnp.all(jnp.isfinite(g))
        assert jnp.all(g < 0)  # more absorption → less signal


# ---------------------------------------------------------------------------
# Jacobian w.r.t. absorption
# ---------------------------------------------------------------------------

class TestComputeJacobianMua:
    def test_shape(self, box_mesh):
        """Jacobian should be (n_meas, nn) where n_meas = n_det * n_src."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])

        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        n_meas = 1 * 1  # n_det * n_src
        assert J.shape == (n_meas, box_mesh.nn)

    def test_finite(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        assert jnp.all(jnp.isfinite(J))

    def test_negative_entries(self, box_mesh):
        """Jacobian d(detval)/d(mua_node) should be mostly negative."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        # More absorption at any node → less signal, so Jacobian should be <= 0
        assert jnp.sum(J < 0) > 0

    def test_multi_measurement(self, box_mesh):
        srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
        detpos = jnp.array([[10.0, 10.0, 10.0]])
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        assert J.shape == (2, box_mesh.nn)  # 1 det * 2 src = 2 measurements
