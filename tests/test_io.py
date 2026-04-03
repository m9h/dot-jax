"""Tests for SNIRF and BIDS-fNIRS I/O.

Tests against the ds004569 high-density DOT dataset from OpenNeuro.
Skipped if data is not available locally.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

from dot_jax.io import (
    read_snirf,
    load_bids_nirs,
    snirf_to_dot_jax,
    SnirfData,
    BidsNirsRun,
)


DS004569 = Path("/data/raw/ds004569")
SNIRF_FILE = DS004569 / "sub-01/ses-01/nirs/sub-01_ses-01_task-movie.snirf"
NIRS_DIR = DS004569 / "sub-01/ses-01/nirs"

has_ds004569 = SNIRF_FILE.exists() and SNIRF_FILE.stat().st_size > 1000


# ---------------------------------------------------------------------------
# read_snirf
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_ds004569, reason="ds004569 not downloaded")
class TestReadSnirf:
    @pytest.fixture(scope="class")
    def snirf(self):
        return read_snirf(SNIRF_FILE)

    def test_returns_snirf_data(self, snirf):
        assert isinstance(snirf, SnirfData)

    def test_data_shape(self, snirf):
        """ds004569 sub-01 ses-01: 6722 timepoints × 3684 channels."""
        assert snirf.data.ndim == 2
        assert snirf.data.shape[1] == 3684
        assert snirf.data.shape[0] > 1000  # >1000 timepoints

    def test_data_finite(self, snirf):
        # At least 99% of data should be finite (allow some NaN/Inf in bad channels)
        finite_frac = np.isfinite(snirf.data).mean()
        assert finite_frac > 0.99

    def test_source_positions(self, snirf):
        """96 sources in 3D Talairach coordinates."""
        assert snirf.source_pos.shape == (96, 3)
        assert np.all(np.isfinite(snirf.source_pos))

    def test_detector_positions(self, snirf):
        """92 detectors in 3D."""
        assert snirf.detector_pos.shape == (92, 3)
        assert np.all(np.isfinite(snirf.detector_pos))

    def test_wavelengths(self, snirf):
        """750nm and 850nm."""
        npt.assert_array_equal(snirf.wavelengths, [750.0, 850.0])

    def test_measurement_list_length(self, snirf):
        """3684 channels total."""
        assert len(snirf.measurement_list.source_index) == 3684
        assert len(snirf.measurement_list.detector_index) == 3684
        assert len(snirf.measurement_list.wavelength_index) == 3684

    def test_measurement_list_indices_1based(self, snirf):
        """SNIRF uses 1-based indices."""
        ml = snirf.measurement_list
        assert ml.source_index.min() >= 1
        assert ml.source_index.max() <= 96
        assert ml.detector_index.min() >= 1
        assert ml.detector_index.max() <= 92
        assert ml.wavelength_index.min() >= 1
        assert ml.wavelength_index.max() <= 2

    def test_sampling_frequency(self, snirf):
        """~10 Hz sampling rate."""
        assert snirf.sampling_frequency is not None
        assert 9.0 < snirf.sampling_frequency < 11.0

    def test_equal_channels_per_wavelength(self, snirf):
        """Should have equal channels at each wavelength."""
        ml = snirf.measurement_list
        n_wl1 = np.sum(ml.wavelength_index == 1)
        n_wl2 = np.sum(ml.wavelength_index == 2)
        assert n_wl1 == n_wl2 == 1842


# ---------------------------------------------------------------------------
# snirf_to_dot_jax
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_ds004569, reason="ds004569 not downloaded")
class TestSnirfToDotJax:
    @pytest.fixture(scope="class")
    def snirf(self):
        return read_snirf(SNIRF_FILE)

    def test_full_extraction(self, snirf):
        d = snirf_to_dot_jax(snirf)
        assert isinstance(d["srcpos"], jnp.ndarray)
        assert d["srcpos"].shape == (96, 3)
        assert d["detpos"].shape == (92, 3)
        assert d["data"].shape[1] == 3684

    def test_wavelength_filter_750(self, snirf):
        d = snirf_to_dot_jax(snirf, wavelength=750)
        assert d["data"].shape[1] == 1842
        npt.assert_allclose(d["channel_wl"], 750.0, atol=5.0)

    def test_wavelength_filter_850(self, snirf):
        d = snirf_to_dot_jax(snirf, wavelength=850)
        assert d["data"].shape[1] == 1842
        npt.assert_allclose(d["channel_wl"], 850.0, atol=5.0)

    def test_0based_channel_indices(self, snirf):
        d = snirf_to_dot_jax(snirf)
        assert d["channel_src"].min() == 0
        assert d["channel_det"].min() == 0
        assert d["channel_src"].max() == 95  # 96 sources, 0-based
        assert d["channel_det"].max() == 91

    def test_invalid_wavelength_raises(self, snirf):
        with pytest.raises(ValueError, match="No channels"):
            snirf_to_dot_jax(snirf, wavelength=999)

    def test_fs_propagated(self, snirf):
        d = snirf_to_dot_jax(snirf)
        assert d["fs"] is not None
        assert 9.0 < d["fs"] < 11.0


# ---------------------------------------------------------------------------
# load_bids_nirs
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not has_ds004569, reason="ds004569 not downloaded")
class TestLoadBidsNirs:
    def test_load_session(self):
        run = load_bids_nirs(NIRS_DIR, task="movie")
        assert isinstance(run, BidsNirsRun)
        assert isinstance(run.snirf, SnirfData)

    def test_task_name(self):
        run = load_bids_nirs(NIRS_DIR, task="movie")
        assert run.task_name == "Movie"

    def test_metadata(self):
        run = load_bids_nirs(NIRS_DIR, task="movie")
        assert run.metadata is not None
        assert run.metadata["NIRSChannelCount"] == 3684
        assert run.metadata["NIRSSourceOptodeCount"] == 96
        assert run.metadata["NIRSDetectorOptodeCount"] == 92

    def test_sampling_frequency_from_sidecar(self):
        """Sampling frequency should come from _nirs.json sidecar."""
        run = load_bids_nirs(NIRS_DIR, task="movie")
        assert run.snirf.sampling_frequency is not None
        npt.assert_allclose(run.snirf.sampling_frequency, 10.0005, rtol=1e-3)

    def test_events(self):
        run = load_bids_nirs(NIRS_DIR, task="movie")
        assert run.events is not None
        assert run.events.shape[1] == 3
        # onset=30, duration=641
        npt.assert_allclose(run.events[0, 0], 30.0)
        npt.assert_allclose(run.events[0, 1], 641.0)

    def test_optodes_tsv(self):
        run = load_bids_nirs(NIRS_DIR, task="movie")
        assert run.optodes_tsv is not None
        assert "name" in run.optodes_tsv
        assert "x" in run.optodes_tsv
        # 96 sources + 92 detectors = 188 optodes
        assert len(run.optodes_tsv["name"]) == 188

    def test_channels_tsv(self):
        run = load_bids_nirs(NIRS_DIR, task="movie")
        assert run.channels_tsv is not None
        assert len(run.channels_tsv["name"]) == 3684

    def test_coordsystem(self):
        run = load_bids_nirs(NIRS_DIR, task="movie")
        assert run.coordsystem is not None
        assert run.coordsystem["NIRSCoordinateSystem"] == "Talairach"
        assert run.coordsystem["NIRSCoordinateUnits"] == "mm"

    def test_missing_task_raises(self):
        with pytest.raises(FileNotFoundError):
            load_bids_nirs(NIRS_DIR, task="nonexistent")
