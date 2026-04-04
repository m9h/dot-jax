"""Tests for hemodynamic preprocessing — RED-GREEN TDD.

Validates the signal processing chain from raw CW intensity to
chromophore concentration changes (HbO/HbR).
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from dot_jax.hemodynamics import (
    intensity_to_od,
    bandpass_filter,
    downsample,
    spectral_unmix,
)


# ---------------------------------------------------------------------------
# Intensity to optical density
# ---------------------------------------------------------------------------

class TestIntensityToOD:
    def test_shape_preserved(self):
        """OD should have same shape as intensity."""
        I = jnp.ones((100, 50)) * 1000.0
        od = intensity_to_od(I)
        assert od.shape == I.shape

    def test_zero_at_mean(self):
        """Constant intensity → zero OD change."""
        I = jnp.ones((100, 50)) * 500.0
        od = intensity_to_od(I)
        npt.assert_allclose(od, 0.0, atol=1e-12)

    def test_sign_convention(self):
        """Increased absorption (lower I) → positive OD."""
        # Channel with mean=1000: timepoint at 500 should give positive OD
        I = jnp.array([[1000.0], [500.0], [1500.0]])
        od = intensity_to_od(I)
        assert od[1, 0] > 0   # below-mean intensity → positive OD
        assert od[2, 0] < 0   # above-mean intensity → negative OD

    def test_known_values(self):
        """OD = -log(I / I_mean)."""
        I = jnp.array([[100.0, 50.0, 200.0]])
        I_mean = jnp.mean(I, axis=0)
        expected = -jnp.log(I / I_mean)
        od = intensity_to_od(I)
        npt.assert_allclose(od, expected, rtol=1e-12)

    def test_finite(self):
        I = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (200, 30))) + 1.0
        od = intensity_to_od(I)
        assert jnp.all(jnp.isfinite(od))


# ---------------------------------------------------------------------------
# Bandpass filter
# ---------------------------------------------------------------------------

class TestBandpassFilter:
    def test_shape_preserved(self):
        data = jnp.ones((1000, 10))
        filt = bandpass_filter(data, fs=10.0, low=0.01, high=0.5)
        assert filt.shape == data.shape

    def test_removes_dc(self):
        """DC offset should be removed by high-pass."""
        t = jnp.linspace(0, 100, 1000)
        signal = 5.0 + 0.1 * jnp.sin(2 * jnp.pi * 0.1 * t)
        data = signal[:, None]
        filt = bandpass_filter(data, fs=10.0, low=0.01, high=0.5)
        # Mean should be near zero
        assert jnp.abs(jnp.mean(filt)) < 0.5

    def test_preserves_passband(self):
        """Signal within passband should be mostly preserved."""
        fs = 10.0
        t = jnp.arange(0, 100, 1.0 / fs)
        signal = jnp.sin(2 * jnp.pi * 0.1 * t)  # 0.1 Hz, well within band
        data = signal[:, None]
        filt = bandpass_filter(data, fs=fs, low=0.01, high=0.5)
        # Amplitude should be preserved (within ~20% for edge effects)
        assert jnp.std(filt) > 0.5 * jnp.std(data)

    def test_finite(self):
        data = jax.random.normal(jax.random.PRNGKey(1), (500, 20))
        filt = bandpass_filter(data, fs=10.0, low=0.01, high=0.5)
        assert jnp.all(jnp.isfinite(filt))


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------

class TestDownsample:
    def test_shape(self):
        data = jnp.ones((1000, 10))
        ds = downsample(data, factor=10)
        assert ds.shape == (100, 10)

    def test_values(self):
        """Downsampling by averaging should preserve mean."""
        data = jnp.arange(100.0)[:, None]
        ds = downsample(data, factor=10)
        assert ds.shape == (10, 1)
        # First block mean: mean(0..9) = 4.5
        npt.assert_allclose(ds[0, 0], 4.5, atol=1e-10)

    def test_single_channel(self):
        data = jnp.ones((200, 1)) * 3.0
        ds = downsample(data, factor=5)
        assert ds.shape == (40, 1)
        npt.assert_allclose(ds, 3.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Spectral unmixing
# ---------------------------------------------------------------------------

class TestSpectralUnmix:
    def test_shape(self):
        """Should return (n_time, n_nodes, n_chromophores)."""
        from dot_jax.property import extinction
        E = extinction([750, 850], ["hbo", "hbr"])
        # delta_mua: (n_wv, n_time, n_nodes)
        delta_mua = jnp.zeros((2, 100, 50))
        hbo, hbr = spectral_unmix(delta_mua, E)
        assert hbo.shape == (100, 50)
        assert hbr.shape == (100, 50)

    def test_zero_in_zero_out(self):
        from dot_jax.property import extinction
        E = extinction([750, 850], ["hbo", "hbr"])
        delta_mua = jnp.zeros((2, 10, 5))
        hbo, hbr = spectral_unmix(delta_mua, E)
        npt.assert_allclose(hbo, 0.0, atol=1e-15)
        npt.assert_allclose(hbr, 0.0, atol=1e-15)

    def test_round_trip(self):
        """Known concentration → mua → unmix should recover concentrations."""
        from dot_jax.property import extinction
        E = extinction([750, 850], ["hbo", "hbr"])
        # Known chromophore changes
        delta_hbo = jnp.array([[[1.0]]])   # (1, 1) shape
        delta_hbr = jnp.array([[[0.5]]])
        delta_c = jnp.concatenate([delta_hbo, delta_hbr], axis=-1)  # (1, 1, 2)
        # Forward: mua = E @ c
        delta_mua = jnp.einsum("wc,tnc->wtn", E, delta_c)  # (2, 1, 1)
        # Inverse
        hbo, hbr = spectral_unmix(delta_mua, E)
        npt.assert_allclose(hbo[0, 0], 1.0, rtol=1e-10)
        npt.assert_allclose(hbr[0, 0], 0.5, rtol=1e-10)

    def test_finite(self):
        from dot_jax.property import extinction
        E = extinction([750, 850], ["hbo", "hbr"])
        delta_mua = jax.random.normal(jax.random.PRNGKey(2), (2, 50, 20)) * 1e-4
        hbo, hbr = spectral_unmix(delta_mua, E)
        assert jnp.all(jnp.isfinite(hbo))
        assert jnp.all(jnp.isfinite(hbr))
