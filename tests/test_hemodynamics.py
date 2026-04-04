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
    compute_channel_snr,
    prune_channels,
    detect_motion_artifacts,
    correct_motion_spline,
    normalize_od,
    zscore_images,
    compute_gvtd,
    identify_short_channels,
    regress_short_channels,
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


# ---------------------------------------------------------------------------
# Channel quality / SNR pruning
# ---------------------------------------------------------------------------

class TestComputeChannelSnr:
    def test_shape(self):
        I = jnp.ones((100, 5)) * 1000.0 + jax.random.normal(jax.random.PRNGKey(0), (100, 5))
        snr = compute_channel_snr(I)
        assert snr.shape == (5,)

    def test_high_snr_for_constant(self):
        """Constant + tiny noise → very high SNR."""
        I = jnp.ones((1000, 3)) * 1000.0 + 1e-10 * jax.random.normal(jax.random.PRNGKey(0), (1000, 3))
        snr = compute_channel_snr(I)
        assert jnp.all(snr > 1e6)

    def test_low_snr_for_noise(self):
        """Pure noise → SNR ~ 0 (mean near 0)."""
        I = jax.random.normal(jax.random.PRNGKey(1), (1000, 3))
        snr = compute_channel_snr(I)
        assert jnp.all(snr < 5)

    def test_positive(self):
        I = jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (200, 10))) + 1.0
        snr = compute_channel_snr(I)
        assert jnp.all(snr >= 0)


class TestPruneChannels:
    def test_shape(self):
        data = jnp.ones((100, 10)) * 1000.0
        mask = prune_channels(data)
        assert mask.shape == (10,)
        assert mask.dtype == bool

    def test_good_channels_pass(self):
        """Clean channels should all pass."""
        data = jnp.ones((100, 5)) * 500.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (100, 5))
        mask = prune_channels(data, snr_threshold=2.0)
        assert jnp.all(mask)

    def test_noisy_channel_rejected(self):
        """Channel with zero mean and high variance should be rejected."""
        data = jnp.ones((100, 3)) * 500.0
        noise = jax.random.normal(jax.random.PRNGKey(0), (100, 1)) * 1000.0
        data = data.at[:, 2].set(noise[:, 0])
        mask = prune_channels(data, snr_threshold=2.0)
        assert not mask[2]

    def test_saturated_channel_rejected(self):
        """Channel stuck at one value should be rejected."""
        data = jnp.ones((100, 3)) * 500.0
        data = data.at[:, 1].set(0.0)  # stuck at zero
        mask = prune_channels(data, snr_threshold=0.0)
        # CV of zero-variance channel is 0 or undefined → rejected by sat check
        assert not mask[1]


# ---------------------------------------------------------------------------
# Motion artifact detection and correction
# ---------------------------------------------------------------------------

class TestDetectMotionArtifacts:
    def test_clean_signal_no_artifacts(self):
        """Smooth signal should have no artifacts."""
        t = jnp.linspace(0, 10, 1000)
        data = jnp.sin(2 * jnp.pi * 0.1 * t)[:, None]
        mask = detect_motion_artifacts(data, fs=100.0, std_threshold=5.0)
        assert mask.shape == data.shape
        assert jnp.sum(mask) == 0

    def test_spike_detected(self):
        """A large spike should be flagged."""
        data = jnp.zeros((200, 1))
        data = data.at[100, 0].set(100.0)  # huge spike
        mask = detect_motion_artifacts(data, fs=10.0, std_threshold=3.0)
        # Spike should be detected near sample 100
        assert jnp.any(mask[99:102, 0])

    def test_shape_preserved(self):
        data = jax.random.normal(jax.random.PRNGKey(0), (500, 10)) * 0.01
        mask = detect_motion_artifacts(data, fs=10.0)
        assert mask.shape == data.shape


class TestCorrectMotionSpline:
    def test_no_artifacts_unchanged(self):
        """No artifacts → output unchanged."""
        data = jnp.sin(jnp.linspace(0, 6, 100))[:, None]
        mask = jnp.zeros((100, 1), dtype=bool)
        corrected = correct_motion_spline(data, mask, fs=10.0)
        npt.assert_allclose(corrected, data, atol=1e-10)

    def test_spike_reduced(self):
        """Corrected spike should be closer to neighbors."""
        data = jnp.zeros((100, 1))
        data = data.at[50, 0].set(10.0)
        mask = jnp.zeros((100, 1), dtype=bool).at[50, 0].set(True)
        corrected = correct_motion_spline(data, mask, fs=10.0)
        assert jnp.abs(corrected[50, 0]) < jnp.abs(data[50, 0])

    def test_shape_preserved(self):
        data = jnp.ones((200, 5))
        mask = jnp.zeros((200, 5), dtype=bool)
        corrected = correct_motion_spline(data, mask, fs=10.0)
        assert corrected.shape == data.shape


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalizeOD:
    def test_shape(self):
        od = jnp.ones((100, 20))
        pred = jnp.ones(20) * 0.5
        normed = normalize_od(od, pred)
        assert normed.shape == od.shape

    def test_scales_inversely(self):
        od = jnp.ones((10, 3))
        normed1 = normalize_od(od, jnp.array([1.0, 1.0, 1.0]))
        normed2 = normalize_od(od, jnp.array([2.0, 2.0, 2.0]))
        npt.assert_allclose(normed2, normed1 / 2.0, rtol=1e-12)

    def test_finite(self):
        od = jax.random.normal(jax.random.PRNGKey(0), (50, 10))
        pred = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (10,))) + 0.01
        normed = normalize_od(od, pred)
        assert jnp.all(jnp.isfinite(normed))


class TestZscoreImages:
    def test_shape(self):
        images = jax.random.normal(jax.random.PRNGKey(0), (100, 50))
        z = zscore_images(images)
        assert z.shape == images.shape

    def test_zero_mean_unit_var(self):
        images = jax.random.normal(jax.random.PRNGKey(0), (1000, 10)) * 5.0 + 3.0
        z = zscore_images(images)
        npt.assert_allclose(jnp.mean(z, axis=0), 0.0, atol=0.05)
        npt.assert_allclose(jnp.std(z, axis=0), 1.0, atol=0.05)

    def test_constant_column_zero(self):
        """Constant column → zero (avoid NaN)."""
        images = jnp.ones((100, 3))
        z = zscore_images(images)
        assert jnp.all(jnp.isfinite(z))
        npt.assert_allclose(z, 0.0, atol=1e-12)


class TestComputeGVTD:
    def test_shape(self):
        data = jax.random.normal(jax.random.PRNGKey(0), (100, 20))
        gvtd = compute_gvtd(data, fs=10.0)
        assert gvtd.shape == (99,)

    def test_positive(self):
        data = jax.random.normal(jax.random.PRNGKey(0), (100, 20))
        gvtd = compute_gvtd(data, fs=10.0)
        assert jnp.all(gvtd >= 0)

    def test_constant_zero(self):
        """Constant data → zero GVTD."""
        data = jnp.ones((50, 10))
        gvtd = compute_gvtd(data, fs=10.0)
        npt.assert_allclose(gvtd, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Short-separation regression
# ---------------------------------------------------------------------------

class TestIdentifyShortChannels:
    def test_known_geometry(self):
        srcpos = jnp.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        detpos = jnp.array([[5.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        ch_src = jnp.array([0, 0, 1, 1])
        ch_det = jnp.array([0, 1, 0, 1])
        mask = identify_short_channels(srcpos, detpos, ch_src, ch_det, max_distance=10.0)
        assert mask[0]       # src0-det0 = 5mm → short
        assert not mask[1]   # src0-det1 = 50mm → long
        assert not mask[2]   # src1-det0 = 95mm → long
        assert not mask[3]   # src1-det1 = 50mm → long

    def test_all_short(self):
        srcpos = jnp.array([[0.0, 0.0, 0.0]])
        detpos = jnp.array([[1.0, 0.0, 0.0]])
        ch_src = jnp.array([0])
        ch_det = jnp.array([0])
        mask = identify_short_channels(srcpos, detpos, ch_src, ch_det, max_distance=10.0)
        assert mask[0]

    def test_shape(self):
        srcpos = jnp.zeros((5, 3))
        detpos = jnp.zeros((5, 3))
        ch_src = jnp.arange(10) % 5
        ch_det = jnp.arange(10) % 5
        mask = identify_short_channels(srcpos, detpos, ch_src, ch_det)
        assert mask.shape == (10,)


class TestRegressShortChannels:
    def test_shape_preserved(self):
        data = jnp.ones((100, 10))
        short_mask = jnp.array([True, False, False, False, False,
                                True, False, False, False, False])
        corrected = regress_short_channels(data, short_mask)
        assert corrected.shape == data.shape

    def test_removes_correlated_signal(self):
        """Superficial signal correlated with short channels should be reduced."""
        np.random.seed(42)
        n_time = 500
        systemic = np.sin(np.linspace(0, 10, n_time))
        brain = np.random.randn(n_time) * 0.1
        short_ch = systemic + np.random.randn(n_time) * 0.01
        long_ch = systemic + brain
        data = jnp.column_stack([short_ch, long_ch])
        short_mask = jnp.array([True, False])
        corrected = regress_short_channels(data, short_mask)
        # After regression, long channel should have less systemic contamination
        corr_before = np.abs(np.corrcoef(systemic, np.array(data[:, 1]))[0, 1])
        corr_after = np.abs(np.corrcoef(systemic, np.array(corrected[:, 1]))[0, 1])
        assert corr_after < corr_before

    def test_no_short_channels_unchanged(self):
        """If no short channels, data should be unchanged."""
        data = jax.random.normal(jax.random.PRNGKey(0), (100, 5))
        short_mask = jnp.zeros(5, dtype=bool)
        corrected = regress_short_channels(data, short_mask)
        npt.assert_allclose(corrected, data, atol=1e-12)
