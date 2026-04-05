"""Tests for streaming receiver and web dashboard — RED-GREEN TDD.

Validates ZMQ frame receiver, SNIRF replay streamer, and websocket
dashboard server for real-time fNIRS visualization.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import time
import threading
import json

jax.config.update("jax_enable_x64", True)

from dot_jax.streaming import (
    FramePacket,
    SNIRFReplayStreamer,
    ZMQFrameReceiver,
    DashboardServer,
)


# ---------------------------------------------------------------------------
# FramePacket
# ---------------------------------------------------------------------------

class TestFramePacket:
    def test_create(self):
        pkt = FramePacket(
            timestamp=1.0,
            data=np.ones((100,), dtype=np.float64),
            wavelength_indices=np.zeros(100, dtype=np.int32),
            source_indices=np.zeros(100, dtype=np.int32),
            detector_indices=np.zeros(100, dtype=np.int32),
            event_type=None,
        )
        assert pkt.timestamp == 1.0
        assert pkt.data.shape == (100,)

    def test_serialize_deserialize(self):
        """Pack and unpack should round-trip."""
        pkt = FramePacket(
            timestamp=2.5,
            data=np.array([1.0, 2.0, 3.0]),
            wavelength_indices=np.array([0, 0, 1]),
            source_indices=np.array([0, 1, 0]),
            detector_indices=np.array([0, 0, 1]),
            event_type="stimulus",
        )
        buf = pkt.serialize()
        assert isinstance(buf, bytes)
        pkt2 = FramePacket.deserialize(buf)
        assert pkt2.timestamp == pytest.approx(2.5)
        npt.assert_array_equal(pkt2.data, pkt.data)
        assert pkt2.event_type == "stimulus"


# ---------------------------------------------------------------------------
# SNIRF Replay Streamer
# ---------------------------------------------------------------------------

class TestSNIRFReplayStreamer:
    def test_init_from_snirf(self):
        """Should init from a real or mock SNIRF file."""
        # Create minimal mock data
        n_time, n_ch = 50, 10
        data = np.random.rand(n_time, n_ch).astype(np.float64)
        wl = np.array([690.0] * 5 + [830.0] * 5)
        src = np.arange(n_ch) % 3
        det = np.arange(n_ch) % 4
        streamer = SNIRFReplayStreamer.from_arrays(
            data=data, fs=10.0,
            wavelength_values=wl,
            source_indices=src,
            detector_indices=det,
        )
        assert streamer.n_frames == 50
        assert streamer.n_channels == 10
        assert streamer.fs == 10.0

    def test_iterate_frames(self):
        n_time, n_ch = 20, 5
        data = np.ones((n_time, n_ch))
        streamer = SNIRFReplayStreamer.from_arrays(
            data=data, fs=10.0,
            wavelength_values=np.zeros(n_ch),
            source_indices=np.zeros(n_ch, dtype=int),
            detector_indices=np.zeros(n_ch, dtype=int),
        )
        frames = list(streamer.iter_frames())
        assert len(frames) == 20
        assert isinstance(frames[0], FramePacket)

    def test_respects_timing(self):
        """In real-time mode, frames should be spaced by 1/fs."""
        data = np.ones((5, 3))
        streamer = SNIRFReplayStreamer.from_arrays(
            data=data, fs=100.0,  # 100 Hz → 10ms per frame
            wavelength_values=np.zeros(3),
            source_indices=np.zeros(3, dtype=int),
            detector_indices=np.zeros(3, dtype=int),
        )
        t0 = time.time()
        frames = list(streamer.iter_frames(realtime=True))
        elapsed = time.time() - t0
        # 5 frames at 100 Hz → ~0.04 s minimum
        assert elapsed >= 0.03
        assert len(frames) == 5

    def test_fast_mode(self):
        """In fast mode, all frames should return immediately."""
        data = np.ones((100, 3))
        streamer = SNIRFReplayStreamer.from_arrays(
            data=data, fs=1.0,
            wavelength_values=np.zeros(3),
            source_indices=np.zeros(3, dtype=int),
            detector_indices=np.zeros(3, dtype=int),
        )
        t0 = time.time()
        frames = list(streamer.iter_frames(realtime=False))
        elapsed = time.time() - t0
        assert elapsed < 1.0  # should be near-instant
        assert len(frames) == 100


# ---------------------------------------------------------------------------
# ZMQ Frame Receiver (integration test — uses localhost)
# ---------------------------------------------------------------------------

zmq_available = False
try:
    import zmq
    zmq_available = True
except ImportError:
    pass


@pytest.mark.skipif(not zmq_available, reason="pyzmq not installed")
class TestZMQFrameReceiver:
    def test_send_receive(self):
        """Send a frame over ZMQ, receive it."""
        import zmq

        port = 15555
        receiver = ZMQFrameReceiver(port=port)

        # Sender in a thread
        pkt = FramePacket(
            timestamp=1.0,
            data=np.array([1.0, 2.0]),
            wavelength_indices=np.array([0, 1]),
            source_indices=np.array([0, 0]),
            detector_indices=np.array([0, 1]),
            event_type=None,
        )

        def sender():
            time.sleep(0.1)
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PUSH)
            sock.connect(f"tcp://localhost:{port}")
            sock.send(pkt.serialize())
            time.sleep(0.1)
            sock.close()
            ctx.term()

        t = threading.Thread(target=sender)
        t.start()

        received = receiver.recv(timeout=2000)
        t.join()
        receiver.close()

        assert received is not None
        assert received.timestamp == pytest.approx(1.0)
        npt.assert_array_equal(received.data, pkt.data)


# ---------------------------------------------------------------------------
# Dashboard Server
# ---------------------------------------------------------------------------

class TestDashboardServer:
    def test_init(self):
        srv = DashboardServer(port=18080)
        assert srv.port == 18080

    def test_format_state(self):
        """Should produce a JSON-serializable state dict."""
        srv = DashboardServer(port=18081)
        state = srv.format_state(
            hbo=np.array([0.1, 0.2, 0.3]),
            hbr=np.array([-0.1, -0.2, -0.3]),
            gvtd=0.005,
            frame_count=42,
            epoch_avg=None,
        )
        # Should be JSON-serializable
        json_str = json.dumps(state)
        assert "hbo" in json_str
        assert "frame_count" in json_str

    def test_format_state_with_epochs(self):
        srv = DashboardServer(port=18082)
        state = srv.format_state(
            hbo=np.array([0.1, 0.2]),
            hbr=np.array([-0.1, -0.2]),
            gvtd=0.005,
            frame_count=100,
            epoch_avg=np.array([[0.1, 0.2], [0.3, 0.4]]),
        )
        parsed = json.loads(json.dumps(state))
        assert "epoch_avg" in parsed
        assert parsed["epoch_avg"] is not None
