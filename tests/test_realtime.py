"""Tests for real-time reconstruction pipeline — RED-GREEN TDD.

Validates the pre-computed fast reconstruction path, streaming
receiver, and event-locked averaging for real-time fNIRS.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
import time

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.realtime import (
    RealtimePipeline,
    EpochAccumulator,
)


@pytest.fixture
def box_mesh():
    node = jnp.array([
        [0.0, 0.0, 0.0], [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0], [20.0, 20.0, 0.0],
        [0.0, 0.0, 20.0], [20.0, 0.0, 20.0],
        [0.0, 20.0, 20.0], [20.0, 20.0, 20.0],
    ], dtype=jnp.float64)
    elem = jnp.array([
        [0, 1, 2, 4], [1, 2, 3, 7], [1, 2, 4, 7],
        [1, 4, 5, 7], [2, 4, 6, 7],
    ], dtype=jnp.int32)
    return FEMMesh.create(node, elem)


# ---------------------------------------------------------------------------
# RealtimePipeline
# ---------------------------------------------------------------------------

class TestRealtimePipeline:
    def test_init(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        pipe = RealtimePipeline(
            box_mesh, srcpos, detpos, mua=0.01, musp=1.0,
            wavelengths=[690.0, 830.0],
        )
        assert pipe.nn == box_mesh.nn
        assert pipe.n_det == 1
        assert pipe.n_src == 1

    def test_jinv_precomputed(self, box_mesh):
        """Jinv should be precomputed at init."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        pipe = RealtimePipeline(
            box_mesh, srcpos, detpos, mua=0.01, musp=1.0,
            wavelengths=[690.0, 830.0],
        )
        assert hasattr(pipe, "Jinv")
        assert len(pipe.Jinv) == 2  # one per wavelength

    def test_process_frame_shape(self, box_mesh):
        """process_frame should return HbO/HbR arrays."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        pipe = RealtimePipeline(
            box_mesh, srcpos, detpos, mua=0.01, musp=1.0,
            wavelengths=[690.0, 830.0],
        )
        # Simulate one frame of OD data per wavelength
        od_frame = {0: jnp.ones(pipe.n_channels[0]) * 0.001,
                    1: jnp.ones(pipe.n_channels[1]) * 0.001}
        hbo, hbr = pipe.process_frame(od_frame)
        assert hbo.shape == (box_mesh.nn,)
        assert hbr.shape == (box_mesh.nn,)

    def test_process_frame_finite(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        pipe = RealtimePipeline(
            box_mesh, srcpos, detpos, mua=0.01, musp=1.0,
            wavelengths=[690.0, 830.0],
        )
        od_frame = {0: jnp.ones(pipe.n_channels[0]) * 0.001,
                    1: jnp.ones(pipe.n_channels[1]) * 0.001}
        hbo, hbr = pipe.process_frame(od_frame)
        assert jnp.all(jnp.isfinite(hbo))
        assert jnp.all(jnp.isfinite(hbr))

    def test_process_frame_fast(self, box_mesh):
        """Single frame should process in < 100ms."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        pipe = RealtimePipeline(
            box_mesh, srcpos, detpos, mua=0.01, musp=1.0,
            wavelengths=[690.0, 830.0],
        )
        od_frame = {0: jnp.ones(pipe.n_channels[0]) * 0.001,
                    1: jnp.ones(pipe.n_channels[1]) * 0.001}
        # Warm up JIT
        pipe.process_frame(od_frame)
        # Time it
        t0 = time.time()
        for _ in range(10):
            pipe.process_frame(od_frame)
        dt = (time.time() - t0) / 10
        assert dt < 0.1, f"Frame processing took {dt:.3f}s (>100ms)"

    def test_quality_metrics(self, box_mesh):
        """Pipeline should report GVTD and channel SNR."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        pipe = RealtimePipeline(
            box_mesh, srcpos, detpos, mua=0.01, musp=1.0,
            wavelengths=[690.0, 830.0],
        )
        od_frame = {0: jnp.ones(pipe.n_channels[0]) * 0.001,
                    1: jnp.ones(pipe.n_channels[1]) * 0.001}
        pipe.process_frame(od_frame)
        metrics = pipe.get_quality_metrics()
        assert "frame_count" in metrics
        assert metrics["frame_count"] == 1


# ---------------------------------------------------------------------------
# EpochAccumulator
# ---------------------------------------------------------------------------

class TestEpochAccumulator:
    def test_init(self):
        acc = EpochAccumulator(n_nodes=10, epoch_samples=30, fs=1.0)
        assert acc.n_epochs == 0

    def test_add_event(self):
        acc = EpochAccumulator(n_nodes=10, epoch_samples=5, fs=1.0)
        # Feed 10 frames, mark event at frame 3
        for i in range(10):
            hbo = jnp.ones(10) * float(i)
            acc.add_frame(hbo, event=(i == 3))
        assert acc.n_epochs == 1

    def test_get_average_shape(self):
        acc = EpochAccumulator(n_nodes=10, epoch_samples=5, fs=1.0)
        for i in range(20):
            hbo = jnp.ones(10) * float(i)
            acc.add_frame(hbo, event=(i == 3 or i == 10))
        avg, std = acc.get_average()
        assert avg.shape == (5, 10)  # (epoch_samples, n_nodes)
        assert std.shape == (5, 10)

    def test_average_correct(self):
        """Average of constant epochs should be that constant."""
        acc = EpochAccumulator(n_nodes=2, epoch_samples=3, fs=1.0)
        # Two events, each followed by [1, 1, 1]
        for i in range(10):
            hbo = jnp.ones(2) * 1.0
            acc.add_frame(hbo, event=(i == 0 or i == 5))
        avg, std = acc.get_average()
        npt.assert_allclose(avg, 1.0, atol=1e-10)
        npt.assert_allclose(std, 0.0, atol=1e-10)

    def test_no_events_returns_none(self):
        acc = EpochAccumulator(n_nodes=10, epoch_samples=5, fs=1.0)
        for i in range(5):
            acc.add_frame(jnp.zeros(10))
        result = acc.get_average()
        assert result is None
