"""Tests for Gauss-Newton reconstruction — RED-GREEN TDD.

Validates linearised image reconstruction on the coarse box mesh.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

from dot_jax.mesh import FEMMesh
from dot_jax.forward import forward_cw
from dot_jax.recon import reconstruct_mua


@pytest.fixture
def box_mesh():
    """20x20x20 mm box mesh."""
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
def recon_setup(box_mesh):
    """Generate synthetic data with a known mua perturbation."""
    srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
    detpos = jnp.array([[10.0, 10.0, 10.0], [17.0, 3.0, 3.0]])
    musp = 1.0
    mua_true = 0.015   # small perturbation from background
    mua_bg = 0.01

    # Generate "measured" data at true mua
    result_true = forward_cw(box_mesh, mua_true, musp, srcpos, detpos)

    return {
        "srcpos": srcpos,
        "detpos": detpos,
        "musp": musp,
        "mua_true": mua_true,
        "mua_bg": mua_bg,
        "data": result_true.detval,
    }


class TestReconstructMua:
    def test_returns_recon_result(self, box_mesh, recon_setup):
        from dot_jax import ReconResult
        result = reconstruct_mua(
            box_mesh,
            recon_setup["data"],
            recon_setup["srcpos"],
            recon_setup["detpos"],
            recon_setup["mua_bg"],
            recon_setup["musp"],
        )
        assert isinstance(result, ReconResult)
        assert result.mua.shape == (box_mesh.nn,)
        assert result.residuals.ndim == 1
        assert result.residuals.shape[0] == 2  # 1 step + final

    def test_residual_decreases_single_step(self, box_mesh, recon_setup):
        """Single GN step should reduce residual."""
        result = reconstruct_mua(
            box_mesh,
            recon_setup["data"],
            recon_setup["srcpos"],
            recon_setup["detpos"],
            recon_setup["mua_bg"],
            recon_setup["musp"],
            max_steps=1,
        )
        # After one linearised update, residual should decrease
        assert result.residuals[-1] < result.residuals[0]

    def test_reconstructed_mua_finite(self, box_mesh, recon_setup):
        result = reconstruct_mua(
            box_mesh,
            recon_setup["data"],
            recon_setup["srcpos"],
            recon_setup["detpos"],
            recon_setup["mua_bg"],
            recon_setup["musp"],
        )
        assert jnp.all(jnp.isfinite(result.mua))
        assert jnp.all(jnp.isfinite(result.musp))
        assert jnp.all(jnp.isfinite(result.residuals))

    def test_no_change_for_matched_data(self, box_mesh):
        """If data matches background, delta should be near zero."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        mua_bg = 0.01
        musp = 1.0

        # Generate data at background
        data = forward_cw(box_mesh, mua_bg, musp, srcpos, detpos).detval

        result = reconstruct_mua(
            box_mesh, data, srcpos, detpos, mua_bg, musp,
        )
        # Reconstructed mua should stay near background
        npt.assert_allclose(result.mua, mua_bg, atol=1e-8)

    def test_multi_step(self, box_mesh, recon_setup):
        """Multi-step should not crash and should produce finite results."""
        result = reconstruct_mua(
            box_mesh,
            recon_setup["data"],
            recon_setup["srcpos"],
            recon_setup["detpos"],
            recon_setup["mua_bg"],
            recon_setup["musp"],
            max_steps=3,
        )
        assert result.residuals.shape[0] == 4  # 3 steps + final
        assert jnp.all(jnp.isfinite(result.mua))
