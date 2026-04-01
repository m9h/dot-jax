"""Tests for _types.py — constants and result containers."""

import jax.numpy as jnp
from dot_jax import C0, R_C0, ForwardResult, ReconResult


class TestConstants:
    def test_c0_speed_of_light(self):
        assert C0 == 299792458000.0

    def test_r_c0_reciprocal(self):
        assert R_C0 == 1.0 / C0

    def test_c0_matches_redbirdpy(self):
        from redbirdpy.forward import C0 as C0_rb
        assert C0 == C0_rb


class TestForwardResult:
    def test_construct(self):
        detval = jnp.ones((3, 2))
        phi = jnp.ones((10, 5))
        result = ForwardResult(detval=detval, phi=phi)
        assert result.detval.shape == (3, 2)
        assert result.phi.shape == (10, 5)

    def test_unpack(self):
        detval = jnp.ones((3, 2))
        phi = jnp.ones((10, 5))
        d, p = ForwardResult(detval=detval, phi=phi)
        assert d.shape == (3, 2)
        assert p.shape == (10, 5)


class TestReconResult:
    def test_construct(self):
        result = ReconResult(
            mua=jnp.array([0.01]),
            musp=jnp.array([1.0]),
            residuals=jnp.array([0.5, 0.1]),
        )
        assert result.mua.shape == (1,)
        assert result.residuals.shape == (2,)
