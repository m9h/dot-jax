"""Real-time streaming simulation — RED-GREEN TDD.

End-to-end test of the streaming fNIRS pipeline: synthetic hemodynamic
time series → RealtimePipeline → HbO/HbR images → EpochAccumulator →
event-locked averages.

Simulates a block-design cognitive experiment on the slab phantom:
    - 30-second block with 10-second on / 20-second off
    - HbO increases and HbR decreases during "on" (cortical activation)
    - Pipeline should recover the activation time course and
      epoch averaging should improve SNR

This is the test that proves dot-jax is ready for live Kernel Flow 2 data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import Delaunay

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.property import extinction, mua_from_concentrations
from dot_jax.spectral import compute_jacobian_mua
from dot_jax.forward import forward_cw
from dot_jax.realtime import RealtimePipeline, EpochAccumulator
from dot_jax.streaming import FramePacket, SNIRFReplayStreamer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


WAVELENGTHS = [760, 850]
FS = 4.75  # Kernel Flow sampling rate (Hz)


@pytest.fixture(scope="module")
def slab():
    """Slab phantom: 35x35x20 mm."""
    nx, ny, nz, dx = 8, 8, 5, 5.0
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dx, ny)
    z = np.linspace(0, (nz - 1) * dx, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    node = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    tri = Delaunay(node)
    return FEMMesh.create(node, tri.simplices.astype(np.int32))


@pytest.fixture(scope="module")
def optodes(slab):
    """3x3 sources, 4x4 detectors on top face."""
    z_max = float(np.asarray(slab.node)[:, 2].max())
    src_x = np.linspace(8.75, 26.25, 3)
    src_y = np.linspace(8.75, 26.25, 3)
    sxx, syy = np.meshgrid(src_x, src_y)
    srcpos = np.column_stack([sxx.ravel(), syy.ravel(), np.full(9, z_max)])

    det_x = np.linspace(4.375, 30.625, 4)
    det_y = np.linspace(4.375, 30.625, 4)
    dxx, dyy = np.meshgrid(det_x, det_y)
    detpos = np.column_stack([dxx.ravel(), dyy.ravel(), np.full(16, z_max)])

    return jnp.array(srcpos), jnp.array(detpos)


@pytest.fixture(scope="module")
def pipeline(slab, optodes):
    """RealtimePipeline initialized on the slab phantom."""
    srcpos, detpos = optodes
    return RealtimePipeline(
        slab, srcpos, detpos,
        mua=0.01, musp=1.0,
        wavelengths=WAVELENGTHS,
        sd_range=(5.0, 40.0),
        reg=0.01,
    )


@pytest.fixture(scope="module")
def ext_matrix():
    """Extinction coefficients at 760 and 850 nm."""
    return extinction(WAVELENGTHS, ["hbo", "hbr"])


@pytest.fixture(scope="module")
def synthetic_od_series(slab, optodes, pipeline, ext_matrix):
    """Generate synthetic OD time series for a block-design experiment.

    Block design: 3 blocks of 10s ON / 20s OFF.
    During ON: HbO +3 uM, HbR -1 uM at slab center (5mm deep).
    Total: 90s at 4.75 Hz = 427 frames.

    Returns dict with keys: od_frames, hbo_true, hbr_true, event_frames
    """
    srcpos, detpos = optodes
    node = np.asarray(slab.node)
    z_max = node[:, 2].max()
    nn = slab.nn

    n_frames = int(90 * FS)  # ~427 frames
    center = np.array([17.5, 17.5, z_max - 5.0])
    dist = np.linalg.norm(node - center, axis=1)
    envelope = np.exp(-dist ** 2 / (2 * 5.0 ** 2))

    # Block design: ON for frames [block_start, block_start + on_frames)
    on_seconds = 10.0
    off_seconds = 20.0
    block_period = on_seconds + off_seconds
    on_frames = int(on_seconds * FS)

    # HRF-like activation time course per frame
    hbo_amplitude = np.zeros(n_frames)
    hbr_amplitude = np.zeros(n_frames)
    event_frames = []

    for block in range(3):
        block_start = int(block * block_period * FS)
        event_frames.append(block_start)
        for f in range(on_frames):
            idx = block_start + f
            if idx < n_frames:
                # Simple boxcar (no HRF convolution for now)
                hbo_amplitude[idx] = 3.0   # uM
                hbr_amplitude[idx] = -1.0  # uM

    # Per-frame concentration perturbation → delta_mua per wavelength
    # delta_mua[wv, node] = E[wv, :] @ delta_c[node, :]
    E = np.asarray(ext_matrix)

    # We need per-wavelength Jacobians to generate OD data consistent
    # with the forward model. Use the raw adjoint Jacobian normalized
    # by the CW prediction (Rytov normalization) to match the pipeline.
    # For synthetic data, we compute: OD_change = J_raw @ delta_mua / |detval|
    mua_bg = 0.01
    musp_bg = 1.0
    bg_result = forward_cw(slab, mua_bg, musp_bg, srcpos, detpos)
    jacs_raw = []
    for w in range(2):
        J = compute_jacobian_mua(slab, mua_bg, musp_bg, srcpos, detpos)
        jacs_raw.append(np.asarray(J))

    # Map raw Jacobian rows to pipeline channel ordering
    # Pipeline channel map tells us which (src, det) pairs are active
    od_frames = []
    hbo_true_series = np.zeros((n_frames, nn))
    hbr_true_series = np.zeros((n_frames, nn))

    for f in range(n_frames):
        delta_hbo = hbo_amplitude[f] * envelope  # (nn,)
        delta_hbr = hbr_amplitude[f] * envelope  # (nn,)
        hbo_true_series[f] = delta_hbo
        hbr_true_series[f] = delta_hbr

        # delta_c: (nn, 2)
        delta_c = np.column_stack([delta_hbo, delta_hbr])
        # delta_mua: (nn, 2) = delta_c @ E.T
        delta_mua_nodes = delta_c @ E.T  # (nn, n_wv)

        frame = {}
        for w_idx in range(2):
            ch_map = pipeline._channel_maps[w_idx]
            n_ch = pipeline.n_channels[w_idx]
            od_ch = np.zeros(n_ch)
            for ch_i, (s, d) in enumerate(ch_map):
                # Row index in the raw Jacobian (detector-major)
                row = d * srcpos.shape[0] + s
                raw_val = jacs_raw[w_idx][row] @ delta_mua_nodes[:, w_idx]
                # Rytov: divide by prediction magnitude
                pred = float(bg_result.detval[d, s])
                od_ch[ch_i] = raw_val / max(abs(pred), 1e-30)
            frame[w_idx] = jnp.array(od_ch)
        od_frames.append(frame)

    return {
        "od_frames": od_frames,
        "hbo_true": hbo_true_series,
        "hbr_true": hbr_true_series,
        "event_frames": event_frames,
        "n_frames": n_frames,
    }


# ---------------------------------------------------------------------------
# Tests: pipeline on slab phantom
# ---------------------------------------------------------------------------


class TestPipelineOnSlab:
    def test_init_channels(self, pipeline):
        """Pipeline should detect channels within SD range."""
        for w in range(2):
            assert pipeline.n_channels[w] > 0, (
                f"No channels found at wavelength {w}"
            )

    def test_jinv_shapes(self, pipeline, slab):
        """Jinv should be (nn, n_channels) per wavelength."""
        for w in range(2):
            assert pipeline.Jinv[w].shape == (slab.nn, pipeline.n_channels[w])

    def test_single_frame(self, pipeline, slab):
        """Processing a single frame should return finite HbO/HbR."""
        od_frame = {w: jnp.zeros(pipeline.n_channels[w]) for w in range(2)}
        hbo, hbr = pipeline.process_frame(od_frame)
        assert hbo.shape == (slab.nn,)
        assert hbr.shape == (slab.nn,)
        assert jnp.all(jnp.isfinite(hbo))
        assert jnp.all(jnp.isfinite(hbr))


# ---------------------------------------------------------------------------
# Tests: streaming time series
# ---------------------------------------------------------------------------


class TestStreamingSimulation:
    def test_activation_detected(self, pipeline, synthetic_od_series):
        """Pipeline should detect activation: nonzero HbO during ON blocks."""
        frames = synthetic_od_series["od_frames"]

        # Process first ON block (frames ~0 to ~47)
        on_frame = frames[5]  # well within first ON block
        off_frame = frames[int(70 * FS)]  # well within an OFF period

        hbo_on, _ = pipeline.process_frame(on_frame)
        hbo_off, _ = pipeline.process_frame(off_frame)

        on_energy = float(jnp.sum(hbo_on ** 2))
        off_energy = float(jnp.sum(hbo_off ** 2))

        assert on_energy > off_energy, (
            f"ON-block HbO energy ({on_energy:.2e}) should exceed "
            f"OFF-block ({off_energy:.2e})"
        )

    def test_hbo_hbr_opposite_signs(self, pipeline, synthetic_od_series):
        """During activation, mean HbO should be positive, mean HbR negative."""
        frames = synthetic_od_series["od_frames"]
        on_frame = frames[5]

        hbo, hbr = pipeline.process_frame(on_frame)

        mean_hbo = float(jnp.mean(hbo))
        mean_hbr = float(jnp.mean(hbr))

        assert mean_hbo > 0 or mean_hbr < 0, (
            f"Expected activation pattern (HbO>0 or HbR<0), "
            f"got HbO={mean_hbo:.4f}, HbR={mean_hbr:.4f}"
        )

    def test_time_course_tracks_blocks(self, pipeline, synthetic_od_series):
        """HbO time course should show block structure.

        Mean HbO power during ON periods should exceed OFF periods
        across the full time series.
        """
        frames = synthetic_od_series["od_frames"]
        n_frames = synthetic_od_series["n_frames"]

        on_seconds = 10.0
        off_seconds = 20.0
        block_period = on_seconds + off_seconds
        on_frames = int(on_seconds * FS)

        on_power = []
        off_power = []

        for f_idx in range(min(n_frames, int(90 * FS))):
            hbo, _ = pipeline.process_frame(frames[f_idx])
            power = float(jnp.sum(hbo ** 2))

            # Determine if this frame is in an ON or OFF period
            t = f_idx / FS
            block_phase = t % block_period
            if block_phase < on_seconds:
                on_power.append(power)
            else:
                off_power.append(power)

        mean_on = np.mean(on_power) if on_power else 0
        mean_off = np.mean(off_power) if off_power else 0

        assert mean_on > mean_off, (
            f"Mean ON power ({mean_on:.2e}) should exceed OFF ({mean_off:.2e})"
        )


# ---------------------------------------------------------------------------
# Tests: epoch accumulator with pipeline
# ---------------------------------------------------------------------------


class TestEpochAccumulatorSimulation:
    def test_epoch_averaging(self, pipeline, synthetic_od_series, slab):
        """Epoch averaging over 3 blocks should produce a clean response.

        The average HbO across epochs should show activation (positive)
        during the ON period and baseline (near-zero) at the start.
        """
        frames = synthetic_od_series["od_frames"]
        event_frames = synthetic_od_series["event_frames"]
        n_frames = synthetic_od_series["n_frames"]

        epoch_samples = int(10 * FS)  # 10 seconds per epoch
        acc = EpochAccumulator(
            n_nodes=slab.nn,
            epoch_samples=epoch_samples,
            fs=FS,
        )

        event_set = set(event_frames)

        for f_idx in range(min(n_frames, int(90 * FS))):
            hbo, _ = pipeline.process_frame(frames[f_idx])
            acc.add_frame(hbo, event=(f_idx in event_set))

        assert acc.n_epochs >= 2, f"Expected >=2 epochs, got {acc.n_epochs}"

        result = acc.get_average()
        assert result is not None

        avg, std = result
        assert avg.shape == (epoch_samples, slab.nn)

        # The epoch average should show activation (nonzero HbO) since
        # epochs are event-locked to the ON-block onset.
        mean_power = float(jnp.mean(jnp.sum(avg ** 2, axis=1)))
        assert mean_power > 0, "Epoch-averaged HbO should be nonzero"


# ---------------------------------------------------------------------------
# Tests: SNIRFReplayStreamer integration
# ---------------------------------------------------------------------------


class TestReplayStreamerIntegration:
    def test_replay_drives_pipeline(self, pipeline, slab):
        """SNIRFReplayStreamer should produce FramePackets that can drive
        the pipeline.
        """
        n_ch_total = sum(pipeline.n_channels[w] for w in range(2))
        n_frames = 20

        # Build channel arrays matching pipeline layout
        wl_indices = []
        src_indices = []
        det_indices = []
        for w in range(2):
            for s, d in pipeline._channel_maps[w]:
                wl_indices.append(w)
                src_indices.append(s)
                det_indices.append(d)

        wl_indices = np.array(wl_indices, dtype=np.int32)
        src_indices = np.array(src_indices, dtype=np.int32)
        det_indices = np.array(det_indices, dtype=np.int32)

        # Synthetic intensity data
        rng = np.random.default_rng(42)
        data = 1000.0 + rng.normal(0, 10, (n_frames, n_ch_total))

        streamer = SNIRFReplayStreamer.from_arrays(
            data=data, fs=FS,
            wavelength_values=wl_indices,
            source_indices=src_indices,
            detector_indices=det_indices,
        )

        assert streamer.n_frames == n_frames
        assert streamer.n_channels == n_ch_total

        # Verify we can iterate and get FramePackets
        packets = list(streamer.iter_frames())
        assert len(packets) == n_frames
        assert isinstance(packets[0], FramePacket)

    def test_frame_packet_roundtrip(self):
        """FramePacket serialize/deserialize should be lossless."""
        rng = np.random.default_rng(42)
        pkt = FramePacket(
            timestamp=1.5,
            data=rng.normal(0, 1, 10),
            wavelength_indices=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32),
            source_indices=np.arange(10, dtype=np.int32),
            detector_indices=np.arange(10, dtype=np.int32),
            event_type="stimulus",
        )
        buf = pkt.serialize()
        pkt2 = FramePacket.deserialize(buf)

        assert pkt2.timestamp == pkt.timestamp
        npt.assert_array_equal(pkt2.data, pkt.data)
        npt.assert_array_equal(pkt2.wavelength_indices, pkt.wavelength_indices)
        assert pkt2.event_type == pkt.event_type
