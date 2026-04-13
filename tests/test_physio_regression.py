"""Physiological noise regression — RED-GREEN TDD.

Tests for systemic physiology removal from fNIRS/DOT data when
short-separation channels are not available. Implements:
  - Global mean regression (spatial average as systemic regressor)
  - PCA temporal filtering (remove top N components)

These are essential for HD-DOT datasets like ds004569 (WashU) where
all channels are ≥ 10mm separation and systemic contamination
(cardiac, respiratory, Mayer waves) dominates the signal.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from dot_jax.hemodynamics import bandpass_filter


# ---------------------------------------------------------------------------
# Fixtures: synthetic data with known systemic + neural components
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_fnirs():
    """Synthetic fNIRS data with separable systemic and neural components.

    100 channels, 600 timepoints at 10Hz (60s).
    - Systemic: 0.1Hz Mayer wave + 1Hz cardiac (global, all channels)
    - Neural: block response at 5 channels only (focal activation)
    - Noise: white noise at each channel
    """
    rng = np.random.default_rng(42)
    n_time, n_ch = 600, 100
    fs = 10.0
    t = np.arange(n_time) / fs

    # Systemic physiology (shared across all channels)
    mayer = 0.02 * np.sin(2 * np.pi * 0.1 * t)       # ~0.1 Hz
    cardiac = 0.005 * np.sin(2 * np.pi * 1.0 * t)     # ~1 Hz
    respiratory = 0.01 * np.sin(2 * np.pi * 0.25 * t)  # ~0.25 Hz
    systemic = mayer + cardiac + respiratory  # (n_time,)

    # Channel-specific systemic amplitude (slight variation)
    systemic_weights = 1.0 + 0.1 * rng.normal(0, 1, n_ch)
    systemic_signal = systemic[:, None] * systemic_weights[None, :]  # (n_time, n_ch)

    # Neural signal: block design at channels 10-14 only
    neural = np.zeros((n_time, n_ch))
    # Block 1: 10-20s, Block 2: 35-45s
    block1 = (t >= 10) & (t < 20)
    block2 = (t >= 35) & (t < 45)
    # Simple boxcar convolved with HRF-like exponential
    hrf_t = np.arange(0, 10, 1 / fs)
    hrf = hrf_t * np.exp(-hrf_t / 2)
    hrf /= hrf.max()

    stim = np.zeros(n_time)
    stim[block1] = 1.0
    stim[block2] = 1.0
    response = np.convolve(stim, hrf)[:n_time]
    neural[:, 10:15] = 0.005 * response[:, None]  # 5 channels with activation

    # White noise
    noise = 0.002 * rng.normal(0, 1, (n_time, n_ch))

    # Combined signal
    data = systemic_signal + neural + noise

    return {
        "data": data,
        "systemic": systemic_signal,
        "neural": neural,
        "noise": noise,
        "fs": fs,
        "n_time": n_time,
        "n_ch": n_ch,
        "active_channels": np.arange(10, 15),
    }


# ---------------------------------------------------------------------------
# Tests: global mean regression
# ---------------------------------------------------------------------------

class TestGlobalMeanRegression:

    def test_import(self):
        from dot_jax.hemodynamics import regress_global_mean

    def test_shape_preserved(self, synthetic_fnirs):
        from dot_jax.hemodynamics import regress_global_mean
        data = synthetic_fnirs["data"]
        cleaned = regress_global_mean(data)
        assert cleaned.shape == data.shape

    def test_reduces_global_variance(self, synthetic_fnirs):
        """After regression, the spatial mean should be near zero."""
        from dot_jax.hemodynamics import regress_global_mean
        data = synthetic_fnirs["data"]
        cleaned = regress_global_mean(data)

        global_before = data.mean(axis=1).std()
        global_after = cleaned.mean(axis=1).std()

        assert global_after < 0.1 * global_before, (
            f"Global signal std should drop >90%: {global_before:.4f} → {global_after:.4f}"
        )

    def test_preserves_focal_signal(self, synthetic_fnirs):
        """Focal neural activation should survive global regression."""
        from dot_jax.hemodynamics import regress_global_mean
        exp = synthetic_fnirs
        cleaned = regress_global_mean(exp["data"])

        active = exp["active_channels"]
        inactive = np.setdiff1d(np.arange(exp["n_ch"]), active)

        # Active channels should have more variance than inactive
        var_active = cleaned[:, active].var(axis=0).mean()
        var_inactive = cleaned[:, inactive].var(axis=0).mean()

        assert var_active > var_inactive, (
            f"Active channel variance ({var_active:.6f}) should exceed "
            f"inactive ({var_inactive:.6f})"
        )


# ---------------------------------------------------------------------------
# Tests: PCA temporal filtering
# ---------------------------------------------------------------------------

class TestPCAFilter:

    def test_import(self):
        from dot_jax.hemodynamics import pca_filter

    def test_shape_preserved(self, synthetic_fnirs):
        from dot_jax.hemodynamics import pca_filter
        data = synthetic_fnirs["data"]
        cleaned = pca_filter(data, n_components=3)
        assert cleaned.shape == data.shape

    def test_removes_global_components(self, synthetic_fnirs):
        """Removing top 3 PCs should reduce systemic contamination."""
        from dot_jax.hemodynamics import pca_filter
        exp = synthetic_fnirs

        cleaned = pca_filter(exp["data"], n_components=3)

        # Correlation with true systemic should decrease
        sys_mean = exp["systemic"].mean(axis=1)
        r_before = abs(np.corrcoef(exp["data"].mean(axis=1), sys_mean)[0, 1])
        r_after = abs(np.corrcoef(cleaned.mean(axis=1), sys_mean)[0, 1])

        assert r_after < r_before, (
            f"Systemic correlation should decrease: {r_before:.3f} → {r_after:.3f}"
        )

    def test_preserves_focal_signal(self, synthetic_fnirs):
        """Focal activation (5/100 channels) should survive conservative PCA."""
        from dot_jax.hemodynamics import pca_filter
        exp = synthetic_fnirs

        # Conservative: remove only the dominant global PC
        cleaned = pca_filter(exp["data"], n_components=1)

        active = exp["active_channels"]
        # Active channels should retain more variance than inactive
        var_active = np.asarray(cleaned)[:, active].var(axis=0).mean()
        var_inactive = np.asarray(cleaned)[:, np.setdiff1d(
            np.arange(exp["n_ch"]), active)].var(axis=0).mean()
        assert var_active > var_inactive, (
            f"Active variance ({var_active:.6f}) should exceed inactive ({var_inactive:.6f})"
        )

    def test_n_components_controls_aggressiveness(self, synthetic_fnirs):
        """More components removed → more variance removed."""
        from dot_jax.hemodynamics import pca_filter
        data = synthetic_fnirs["data"]

        cleaned_1 = pca_filter(data, n_components=1)
        cleaned_5 = pca_filter(data, n_components=5)

        assert cleaned_5.var() < cleaned_1.var(), (
            "Removing more PCs should reduce total variance"
        )


# ---------------------------------------------------------------------------
# Tests: combined pipeline on synthetic data
# ---------------------------------------------------------------------------

class TestCombinedDenoising:

    def test_global_regression_improves_neural_cnr(self, synthetic_fnirs):
        """Global regression alone should improve neural contrast-to-noise.

        The systemic signal dominates both active and inactive channels
        equally, so removing it should reveal the focal neural signal
        at active channels while reducing inactive channel variance.
        """
        from dot_jax.hemodynamics import regress_global_mean
        exp = synthetic_fnirs

        cleaned = regress_global_mean(exp["data"])

        active = exp["active_channels"]
        inactive = np.setdiff1d(np.arange(exp["n_ch"]), active)

        # After global regression, inactive channels should lose most
        # variance while active channels retain neural signal
        raw_inactive_var = exp["data"][:, inactive].var(axis=0).mean()
        clean_inactive_var = np.asarray(cleaned)[:, inactive].var(axis=0).mean()

        assert clean_inactive_var < 0.5 * raw_inactive_var, (
            f"Inactive variance should drop >50%: {raw_inactive_var:.6f} → {clean_inactive_var:.6f}"
        )
