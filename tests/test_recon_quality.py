"""Tests for reconstruction quality improvements — RED-GREEN TDD.

Validates depth-weighted Jacobian normalization, L-curve regularization,
and the combined reconstruct_image pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.spectral import compute_jacobian_mua, normalize_jacobian
from dot_jax.recon import reconstruct_image, select_lambda_lcurve
from dot_jax.forward import forward_cw


@pytest.fixture(scope="module")
def box_mesh():
    node = jnp.array([
        [0.0, 0.0, 0.0], [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0], [20.0, 20.0, 0.0],
        [0.0, 0.0, 20.0], [20.0, 0.0, 20.0],
        [0.0, 20.0, 20.0], [20.0, 20.0, 20.0],
    ], dtype=jnp.float64)
    elem = jnp.array([
        [0, 1, 2, 4], [1, 2, 3, 7],
        [1, 2, 4, 7], [1, 4, 5, 7], [2, 4, 6, 7],
    ], dtype=jnp.int32)
    return FEMMesh.create(node, elem)


@pytest.fixture(scope="module")
def box_geometry():
    srcpos = jnp.array([[5.0, 5.0, 5.0], [15.0, 5.0, 5.0]])
    detpos = jnp.array([[5.0, 15.0, 15.0], [15.0, 15.0, 15.0]])
    return srcpos, detpos


# ---------------------------------------------------------------------------
# Depth-weighted Jacobian normalization
# ---------------------------------------------------------------------------

class TestNormalizeJacobian:
    def test_exists(self, box_mesh, box_geometry):
        """normalize_jacobian should be importable and callable."""
        srcpos, detpos = box_geometry
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        J_norm, col_norms = normalize_jacobian(J)
        assert J_norm.shape == J.shape
        assert col_norms.shape == (J.shape[1],)

    def test_unit_column_norms(self, box_mesh, box_geometry):
        """After normalization, each column should have unit L2 norm."""
        srcpos, detpos = box_geometry
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        J_norm, _ = normalize_jacobian(J)
        col_norms_after = jnp.linalg.norm(J_norm, axis=0)
        # Columns with any sensitivity should be unit norm
        active = col_norms_after > 1e-10
        npt.assert_allclose(col_norms_after[active], 1.0, rtol=1e-10)

    def test_recovers_original(self, box_mesh, box_geometry):
        """J_norm * diag(col_norms) should recover original J."""
        srcpos, detpos = box_geometry
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        J_norm, col_norms = normalize_jacobian(J)
        J_recovered = J_norm * col_norms[None, :]
        npt.assert_allclose(J_recovered, J, rtol=1e-10)

    def test_positive_norms(self, box_mesh, box_geometry):
        srcpos, detpos = box_geometry
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        _, col_norms = normalize_jacobian(J)
        assert jnp.all(col_norms >= 0)


# ---------------------------------------------------------------------------
# reconstruct_image (combined pipeline)
# ---------------------------------------------------------------------------

class TestReconstructImage:
    def test_exists(self, box_mesh, box_geometry):
        """reconstruct_image should be importable and callable."""
        srcpos, detpos = box_geometry
        # Generate synthetic data with a perturbation
        r_bg = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        r_pert = forward_cw(box_mesh, 0.015, 1.0, srcpos, detpos)
        delta_data = (r_pert.detval - r_bg.detval).ravel()

        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)
        delta_mua = reconstruct_image(J, delta_data)
        assert delta_mua.shape == (box_mesh.nn,)

    def test_uses_depth_weighting(self, box_mesh, box_geometry):
        """reconstruct_image with depth_weight=True should differ from without."""
        srcpos, detpos = box_geometry
        r_bg = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        r_pert = forward_cw(box_mesh, 0.015, 1.0, srcpos, detpos)
        delta_data = (r_pert.detval - r_bg.detval).ravel()
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)

        x_no_dw = reconstruct_image(J, delta_data, depth_weight=False)
        x_dw = reconstruct_image(J, delta_data, depth_weight=True)
        # They should not be identical
        assert not jnp.allclose(x_no_dw, x_dw)

    def test_reasonable_magnitude(self, box_mesh, box_geometry):
        """Reconstructed delta_mua should have physically reasonable magnitude."""
        srcpos, detpos = box_geometry
        # 50% mua increase: 0.01 → 0.015
        r_bg = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        r_pert = forward_cw(box_mesh, 0.015, 1.0, srcpos, detpos)
        delta_data = (r_pert.detval - r_bg.detval).ravel()
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)

        x = reconstruct_image(J, delta_data, reg_method="lcurve")
        # Mean perturbation was +0.005/mm; reconstructed should be same order
        assert jnp.max(jnp.abs(x)) < 1.0  # not ±200 like GCV gave
        assert jnp.max(jnp.abs(x)) > 1e-6  # not zero either

    def test_lcurve_default(self, box_mesh, box_geometry):
        """Default reg_method should be 'lcurve'."""
        srcpos, detpos = box_geometry
        r_bg = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        r_pert = forward_cw(box_mesh, 0.015, 1.0, srcpos, detpos)
        delta_data = (r_pert.detval - r_bg.detval).ravel()
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)

        # Should not raise
        x = reconstruct_image(J, delta_data)
        assert jnp.all(jnp.isfinite(x))

    def test_fixed_lambda(self, box_mesh, box_geometry):
        """reconstruct_image with fixed reg_param should work."""
        srcpos, detpos = box_geometry
        r_bg = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        r_pert = forward_cw(box_mesh, 0.015, 1.0, srcpos, detpos)
        delta_data = (r_pert.detval - r_bg.detval).ravel()
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)

        x = reconstruct_image(J, delta_data, reg_method="fixed", reg_param=0.01)
        assert jnp.all(jnp.isfinite(x))

    def test_finite(self, box_mesh, box_geometry):
        srcpos, detpos = box_geometry
        r_bg = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        r_pert = forward_cw(box_mesh, 0.015, 1.0, srcpos, detpos)
        delta_data = (r_pert.detval - r_bg.detval).ravel()
        J = compute_jacobian_mua(box_mesh, 0.01, 1.0, srcpos, detpos)

        x = reconstruct_image(J, delta_data)
        assert jnp.all(jnp.isfinite(x))
