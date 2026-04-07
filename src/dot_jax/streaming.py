"""Real-time streaming infrastructure for fNIRS.

Provides frame packet serialisation, SNIRF replay for testing,
ZMQ-based network receiver for live acquisition, and a websocket
dashboard server for real-time visualization.

Architecture:
    Kernel Flow headset → kernel-sdk (laptop) → ZMQ → DGX Spark
        → RealtimePipeline.process_frame() → DashboardServer → browser

Classes:
    FramePacket: Serialisable single-frame data container
    SNIRFReplayStreamer: Replay recorded SNIRF files as frame stream
    ZMQFrameReceiver: Receive frames over ZMQ PULL socket
    DashboardServer: WebSocket server pushing state to browser
"""

import json
import struct
import time
from typing import Optional

import numpy as np


class FramePacket:
    """Single frame of fNIRS data, serialisable for network transport.

    Attributes
    ----------
    timestamp : float — acquisition time (seconds).
    data : (n_channels,) float64 — intensity or OD values.
    wavelength_indices : (n_channels,) int32 — wavelength index per channel.
    source_indices : (n_channels,) int32 — source index per channel.
    detector_indices : (n_channels,) int32 — detector index per channel.
    event_type : str or None — experiment event at this frame.
    """

    def __init__(self, timestamp, data, wavelength_indices,
                 source_indices, detector_indices, event_type=None):
        self.timestamp = float(timestamp)
        self.data = np.asarray(data, dtype=np.float64)
        self.wavelength_indices = np.asarray(wavelength_indices, dtype=np.int32)
        self.source_indices = np.asarray(source_indices, dtype=np.int32)
        self.detector_indices = np.asarray(detector_indices, dtype=np.int32)
        self.event_type = event_type

    def serialize(self):
        """Pack this frame into a binary buffer for network transport.

        The wire format is:
        ``[timestamp(f64) | n_ch(i32) | event_len(i32) |
          data(n_ch*f64) | wl_idx(n_ch*i32) | src_idx(n_ch*i32) |
          det_idx(n_ch*i32) | event_utf8(event_len)]``

        Returns
        -------
        buf : bytes
            Serialised frame.
        """
        n_ch = len(self.data)
        event_bytes = (self.event_type or "").encode("utf-8")
        # Header: timestamp(f64) + n_ch(i32) + event_len(i32)
        header = struct.pack("<diI", self.timestamp, n_ch, len(event_bytes))
        return (
            header
            + self.data.tobytes()
            + self.wavelength_indices.tobytes()
            + self.source_indices.tobytes()
            + self.detector_indices.tobytes()
            + event_bytes
        )

    @classmethod
    def deserialize(cls, buf):
        """Reconstruct a FramePacket from a binary buffer.

        Parameters
        ----------
        buf : bytes
            Buffer produced by :meth:`serialize`.

        Returns
        -------
        FramePacket
        """
        offset = 0
        timestamp, n_ch, event_len = struct.unpack_from("<diI", buf, offset)
        offset += struct.calcsize("<diI")

        data = np.frombuffer(buf, dtype=np.float64, count=n_ch, offset=offset)
        offset += n_ch * 8
        wl = np.frombuffer(buf, dtype=np.int32, count=n_ch, offset=offset)
        offset += n_ch * 4
        src = np.frombuffer(buf, dtype=np.int32, count=n_ch, offset=offset)
        offset += n_ch * 4
        det = np.frombuffer(buf, dtype=np.int32, count=n_ch, offset=offset)
        offset += n_ch * 4
        event = buf[offset:offset + event_len].decode("utf-8") or None

        return cls(timestamp, data.copy(), wl.copy(), src.copy(), det.copy(), event)


class SNIRFReplayStreamer:
    """Replay recorded SNIRF data as a frame stream.

    For testing and demo purposes — replays a recorded session as if
    it were live, with optional real-time pacing.

    Parameters
    ----------
    data : (n_frames, n_channels) — intensity time series.
    fs : float — sampling frequency (Hz).
    wavelength_values : (n_channels,) — wavelength value per channel.
    source_indices : (n_channels,) — source index per channel.
    detector_indices : (n_channels,) — detector index per channel.
    events : list of (frame_idx, event_type) — experiment events.
    """

    def __init__(self, data, fs, wavelength_values, source_indices,
                 detector_indices, events=None):
        self._data = np.asarray(data)
        self.fs = float(fs)
        self._wl = np.asarray(wavelength_values, dtype=np.int32)
        self._src = np.asarray(source_indices, dtype=np.int32)
        self._det = np.asarray(detector_indices, dtype=np.int32)
        self._events = dict(events or [])

    @classmethod
    def from_arrays(cls, data, fs, wavelength_values, source_indices,
                    detector_indices, events=None):
        """Construct a replay streamer from raw arrays.

        Parameters
        ----------
        data : (n_frames, n_channels) array_like
            Intensity time series.
        fs : float
            Sampling frequency (Hz).
        wavelength_values : (n_channels,) array_like
            Wavelength value per channel.
        source_indices : (n_channels,) array_like
            Source index per channel.
        detector_indices : (n_channels,) array_like
            Detector index per channel.
        events : list of (int, str), optional
            Pairs of (frame_index, event_type).

        Returns
        -------
        SNIRFReplayStreamer
        """
        return cls(data, fs, wavelength_values, source_indices,
                   detector_indices, events)

    @classmethod
    def from_snirf(cls, snirf_path, fs=None):
        """Construct a replay streamer from a SNIRF file.

        Parameters
        ----------
        snirf_path : str or Path
            Path to a ``.snirf`` file.
        fs : float, optional
            Override sampling frequency (Hz).  If ``None``, uses the
            value stored in the SNIRF metadata, falling back to 4.75 Hz
            (Kernel Flow default).

        Returns
        -------
        SNIRFReplayStreamer
        """
        from .io import read_snirf, snirf_to_dot_jax
        snirf = read_snirf(str(snirf_path))
        jd = snirf_to_dot_jax(snirf)
        actual_fs = snirf.sampling_frequency or fs or 4.75
        return cls(
            data=np.array(jd["data"]),
            fs=actual_fs,
            wavelength_values=np.array(jd["channel_wl"]),
            source_indices=np.array(jd["channel_src"]).astype(np.int32),
            detector_indices=np.array(jd["channel_det"]).astype(np.int32),
        )

    @property
    def n_frames(self):
        """Total number of frames in the recording."""
        return self._data.shape[0]

    @property
    def n_channels(self):
        """Number of measurement channels."""
        return self._data.shape[1]

    def iter_frames(self, realtime=False):
        """Iterate over frames as FramePackets.

        Parameters
        ----------
        realtime : bool — if True, sleep between frames to match fs.

        Yields
        ------
        FramePacket for each frame.
        """
        dt = 1.0 / self.fs
        t0 = time.time()

        for i in range(self.n_frames):
            event = self._events.get(i)
            yield FramePacket(
                timestamp=i * dt,
                data=self._data[i],
                wavelength_indices=self._wl,
                source_indices=self._src,
                detector_indices=self._det,
                event_type=event,
            )
            if realtime and i < self.n_frames - 1:
                target = t0 + (i + 1) * dt
                sleep_time = target - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)


class ZMQFrameReceiver:
    """Receive FramePackets over a ZMQ PULL socket.

    Listens on tcp://*:{port} for serialised FramePackets pushed
    from the acquisition laptop.

    Parameters
    ----------
    port : int — TCP port to listen on.
    """

    def __init__(self, port=15555):
        import zmq
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PULL)
        self._sock.bind(f"tcp://*:{port}")
        self.port = port

    def recv(self, timeout=1000):
        """Receive a single frame.

        Parameters
        ----------
        timeout : int — timeout in milliseconds.

        Returns
        -------
        FramePacket or None if timeout.
        """
        if self._sock.poll(timeout):
            buf = self._sock.recv()
            return FramePacket.deserialize(buf)
        return None

    def iter_frames(self, timeout=1000):
        """Iterate over incoming frames until timeout with no data.

        Parameters
        ----------
        timeout : int
            Timeout in milliseconds to wait between frames.  The
            iterator terminates when no frame arrives within this
            window.

        Yields
        ------
        FramePacket
        """
        while True:
            pkt = self.recv(timeout=timeout)
            if pkt is None:
                break
            yield pkt

    def close(self):
        """Close the ZMQ socket and terminate the context."""
        self._sock.close()
        self._ctx.term()


class DashboardServer:
    """WebSocket server for real-time fNIRS visualization.

    Pushes JSON state updates to connected browser clients.

    Parameters
    ----------
    port : int — WebSocket port.
    """

    def __init__(self, port=18080):
        self.port = port
        self._clients = set()
        self._server = None

    def format_state(self, hbo, hbr, gvtd, frame_count,
                     epoch_avg=None, epoch_std=None):
        """Format pipeline state as a JSON-serializable dict.

        Parameters
        ----------
        hbo : (n_nodes,) — current HbO values.
        hbr : (n_nodes,) — current HbR values.
        gvtd : float — current GVTD quality metric.
        frame_count : int — total frames processed.
        epoch_avg : (epoch_samples, n_nodes) or None — running epoch average.
        epoch_std : (epoch_samples, n_nodes) or None — running epoch std.

        Returns
        -------
        state : dict — JSON-serializable state.
        """
        state = {
            "frame_count": int(frame_count),
            "gvtd": float(gvtd),
            "hbo": np.asarray(hbo).tolist(),
            "hbr": np.asarray(hbr).tolist(),
            "hbo_mean": float(np.mean(hbo)),
            "hbr_mean": float(np.mean(hbr)),
            "hbo_max": float(np.max(np.abs(hbo))),
        }
        if epoch_avg is not None:
            state["epoch_avg"] = np.asarray(epoch_avg).tolist()
        else:
            state["epoch_avg"] = None
        if epoch_std is not None:
            state["epoch_std"] = np.asarray(epoch_std).tolist()
        return state

    async def _handler(self, websocket):
        """Handle a single WebSocket client connection.

        Registers the client, keeps the connection alive, and
        deregisters on disconnect.  Incoming messages from the client
        are ignored.

        Parameters
        ----------
        websocket : websockets.WebSocketServerProtocol
            The connected client.
        """
        self._clients.add(websocket)
        try:
            async for _ in websocket:
                pass  # client messages ignored
        finally:
            self._clients.discard(websocket)

    async def broadcast(self, state):
        """Broadcast a JSON-serializable state dict to all connected clients.

        Parameters
        ----------
        state : dict
            Pipeline state (e.g. from :meth:`format_state`).
        """
        if not self._clients:
            return
        msg = json.dumps(state)
        import websockets
        websockets.broadcast(self._clients, msg)

    async def start(self):
        """Start the WebSocket server, listening on all interfaces."""
        import websockets
        self._server = await websockets.serve(self._handler, "0.0.0.0", self.port)

    async def stop(self):
        """Gracefully shut down the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
