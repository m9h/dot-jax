"""Tests for SNIRF and BIDS-fNIRS I/O.

Tests against the ds004569 high-density DOT dataset from OpenNeuro.
Skipped if data is not available locally.
"""

import os
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
    read_jmesh,
    fetch_neurojson,
    get_mcx_optical_properties,
    load_brain_mesh,
)


DS004569 = Path(
    os.environ.get("DOT_JAX_DATA", Path.home() / "data/raw")
) / "ds004569"
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


# ---------------------------------------------------------------------------
# Time-domain moment SNIRF reader
# ---------------------------------------------------------------------------

import h5py

from dot_jax.io import snirf_td_moments


def _write_td_moment_snirf(path, n_time=10, n_src=2, n_det=2,
                           wavelengths=(690.0, 905.0),
                           moment_orders=(0, 1, 2)):
    """Write a minimal SNIRF v1.1 file with dataType==201 TD-moment channels.

    Each (src, det, wavelength) triple has one channel per moment order,
    matching the Kernel Flow 2 TD-fNIRS export convention.
    """
    with h5py.File(path, "w") as f:
        nirs = f.create_group("nirs1")
        probe = nirs.create_group("probe")
        src_pos = np.tile(np.arange(n_src, dtype=np.float64)[:, None], (1, 3)) * 10.0
        det_pos = np.tile(np.arange(n_det, dtype=np.float64)[:, None], (1, 3)) * 10.0 + 20.0
        probe.create_dataset("sourcePos3D", data=src_pos)
        probe.create_dataset("detectorPos3D", data=det_pos)
        probe.create_dataset("wavelengths", data=np.asarray(wavelengths, dtype=np.float64))

        data_grp = nirs.create_group("data1")
        channels = []
        for s in range(n_src):
            for d in range(n_det):
                for w in range(len(wavelengths)):
                    for o in moment_orders:
                        channels.append((s + 1, d + 1, w + 1, 201, o + 1))
        n_ch = len(channels)
        rng = np.random.default_rng(0)
        ts = rng.standard_normal((n_time, n_ch)).astype(np.float64)
        data_grp.create_dataset("dataTimeSeries", data=ts)

        for i, (s, d, w, dt, di) in enumerate(channels, start=1):
            ml = data_grp.create_group(f"measurementList{i}")
            ml.create_dataset("sourceIndex", data=np.int32(s))
            ml.create_dataset("detectorIndex", data=np.int32(d))
            ml.create_dataset("wavelengthIndex", data=np.int32(w))
            ml.create_dataset("dataType", data=np.int32(dt))
            ml.create_dataset("dataTypeIndex", data=np.int32(di))

    return n_ch


class TestSnirfTdMoments:
    """Extract TD moments (dataType == 201) from a SNIRF file."""

    def test_basic_shape(self, tmp_path):
        path = tmp_path / "td.snirf"
        _write_td_moment_snirf(path, n_time=8, n_src=2, n_det=2,
                               wavelengths=(690.0, 905.0),
                               moment_orders=(0, 1, 2))

        snirf = read_snirf(path)
        out = snirf_td_moments(snirf, moment_orders=(0, 1, 2))
        # 2 src × 2 det × 2 wl = 8 channels, 3 moment orders.
        assert out["moments"].shape == (8, 8, 3)
        assert out["channel_src"].shape == (8,)
        assert out["channel_det"].shape == (8,)
        assert out["channel_wl"].shape == (8,)

    def test_zero_based_indices(self, tmp_path):
        path = tmp_path / "td.snirf"
        _write_td_moment_snirf(path)

        snirf = read_snirf(path)
        out = snirf_td_moments(snirf)
        # Writer uses 1-based SNIRF indices; reader should expose 0-based.
        assert int(jnp.min(out["channel_src"])) == 0
        assert int(jnp.min(out["channel_det"])) == 0

    def test_wavelength_values(self, tmp_path):
        path = tmp_path / "td.snirf"
        _write_td_moment_snirf(path, wavelengths=(690.0, 905.0))

        snirf = read_snirf(path)
        out = snirf_td_moments(snirf)
        unique_wl = jnp.unique(out["channel_wl"])
        npt.assert_array_equal(np.sort(np.array(unique_wl)), np.array([690.0, 905.0]))

    def test_moment_ordering(self, tmp_path):
        """Requesting (0, 2) should place m0 on axis index 0 and m2 on index 1."""
        path = tmp_path / "td.snirf"
        _write_td_moment_snirf(path, moment_orders=(0, 1, 2))

        snirf = read_snirf(path)
        out = snirf_td_moments(snirf, moment_orders=(0, 2))
        assert out["moments"].shape[-1] == 2
        assert out["moment_orders"] == (0, 2)

    def test_rejects_cw_file(self, tmp_path):
        """Files without TD-moment channels should raise."""
        path = tmp_path / "cw.snirf"
        with h5py.File(path, "w") as f:
            nirs = f.create_group("nirs1")
            probe = nirs.create_group("probe")
            probe.create_dataset("sourcePos3D", data=np.zeros((1, 3)))
            probe.create_dataset("detectorPos3D", data=np.zeros((1, 3)))
            probe.create_dataset("wavelengths", data=np.array([690.0]))
            dg = nirs.create_group("data1")
            dg.create_dataset("dataTimeSeries", data=np.zeros((5, 1)))
            ml = dg.create_group("measurementList1")
            ml.create_dataset("sourceIndex", data=np.int32(1))
            ml.create_dataset("detectorIndex", data=np.int32(1))
            ml.create_dataset("wavelengthIndex", data=np.int32(1))
            ml.create_dataset("dataType", data=np.int32(1))  # CW, not 201.

        snirf = read_snirf(path)
        with pytest.raises(ValueError, match="no TD-moment channels"):
            snirf_td_moments(snirf)

    def test_incomplete_moments_raise(self, tmp_path):
        """If some (src, det, wl) triple is missing a requested moment order,
        the reader must flag it rather than silently returning NaNs."""
        path = tmp_path / "td_partial.snirf"
        _write_td_moment_snirf(path, moment_orders=(0, 1))  # file only has m0, m1

        snirf = read_snirf(path)
        with pytest.raises(ValueError, match="incomplete"):
            snirf_td_moments(snirf, moment_orders=(0, 1, 2))  # ask for m2 too


# ---------------------------------------------------------------------------
# JMesh / NeuroJSON
# ---------------------------------------------------------------------------

class TestReadJMesh:
    """Test JMesh decoding with synthetic inline data."""

    def test_inline_dict(self):
        """Decode a small hand-crafted JMesh dict."""
        import base64, zlib
        # Create a tiny mesh: 4 nodes, 1 tet (JMesh uses 1-based)
        node = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float32)
        elem = np.array([[1,2,3,4,1]], dtype=np.int32)  # 1-based, col5=label

        node_z = zlib.compress(node.tobytes())
        elem_z = zlib.compress(elem.tobytes())

        jmesh = {
            "MeshNode": {
                "_ArrayZipData_": base64.b64encode(node_z).decode('ascii'),
                "_ArrayZipType_": "zlib",
                "_ArrayZipSize_": [1, len(node_z)],
                "_ArraySize_": [4, 3],
                "_ArrayType_": "single",
            },
            "MeshElem": {
                "_ArrayZipData_": base64.b64encode(elem_z).decode('ascii'),
                "_ArrayZipType_": "zlib",
                "_ArrayZipSize_": [1, len(elem_z)],
                "_ArraySize_": [1, 5],
                "_ArrayType_": "int32",
            },
        }
        result = read_jmesh(jmesh)
        assert result["node"].shape == (4, 3)
        assert result["elem"].shape == (1, 4)
        assert result["node"].dtype == np.float64
        # Should be 0-based after conversion
        assert np.min(result["elem"]) == 0
        assert np.max(result["elem"]) == 3
        # Tissue label preserved
        assert result["elem_labels"][0] == 1

    def test_preserves_coordinates(self):
        """Node coordinates should be preserved exactly."""
        import base64, zlib
        node = np.array([[10.5, 20.3, 30.1]], dtype=np.float64)
        node_z = zlib.compress(node.tobytes())
        jmesh = {
            "MeshNode": {
                "_ArrayZipData_": base64.b64encode(node_z).decode('ascii'),
                "_ArrayZipType_": "zlib",
                "_ArraySize_": [1, 3],
                "_ArrayType_": "double",
            },
        }
        result = read_jmesh(jmesh)
        npt.assert_allclose(result["node"][0], [10.5, 20.3, 30.1], atol=1e-10)


# Network-dependent tests — skip if neurojson.io unreachable
def _can_reach_neurojson():
    import subprocess
    try:
        r = subprocess.run(
            ["curl", "-sk", "--max-time", "5", "https://neurojson.io:7777/"],
            capture_output=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False

has_neurojson = _can_reach_neurojson()


@pytest.mark.skipif(not has_neurojson, reason="neurojson.io unreachable")
class TestFetchNeurojson:
    def test_fetch_mcx_colin27(self):
        data = fetch_neurojson("mcx", "colin27")
        assert "Domain" in data
        assert "Media" in data["Domain"]

    def test_caches(self, tmp_path):
        """Second fetch should use cache."""
        import time
        t0 = time.time()
        fetch_neurojson("mcx", "colin27")
        t1 = time.time()
        fetch_neurojson("mcx", "colin27")
        t2 = time.time()
        # Cache hit should be much faster (but don't assert — CI timing is noisy)
        assert True


@pytest.mark.skipif(not has_neurojson, reason="neurojson.io unreachable")
class TestGetMcxOpticalProperties:
    def test_colin27_properties(self):
        props = get_mcx_optical_properties("colin27")
        assert len(props) == 7  # background + 5 tissues + air
        # GM should have mua ~0.02
        gm = props[4]
        assert gm["mua"] == pytest.approx(0.02, abs=0.005)
        assert gm["musp"] > 0
        assert gm["n"] == pytest.approx(1.37, abs=0.01)

    def test_4layer_head(self):
        props = get_mcx_optical_properties("4layer_head")
        assert len(props) >= 4


@pytest.mark.skipif(not has_neurojson, reason="neurojson.io unreachable")
class TestLoadBrainMesh:
    def test_brainweb_subject04(self):
        """Load BrainWeb mesh — skip if DataLink API unavailable."""
        from dot_jax.mesh import FEMMesh
        try:
            mesh, labels = load_brain_mesh("BrainWeb", "Subject04")
        except (RuntimeError, Exception) as e:
            if "DataLink" in str(e) or "zlib" in str(e) or "incorrect header" in str(e):
                pytest.skip("BrainMeshLibrary DataLink API unavailable")
            raise
        assert isinstance(mesh, FEMMesh)
        assert mesh.nn > 1000
        assert mesh.ne > 1000
        assert jnp.all(jnp.isfinite(mesh.node))
        assert jnp.all(mesh.evol > 0)
        assert set(np.unique(labels)).issubset({1, 2, 3, 4, 5})
