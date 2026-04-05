#!/usr/bin/env python
"""Demo 7: Real-Time fNIRS — Kernel Flow → DGX Spark → Browser.

Replays Kernel Flow SNIRF data through the dot-jax real-time pipeline
with a web dashboard for live visualization. For the Global NeuroHack.

Usage:
    # Terminal 1: Start the pipeline
    python examples/07_realtime_demo.py

    # Terminal 2: Open browser
    open http://localhost:18080

    # For live acquisition (replace replay with ZMQ receiver):
    python examples/07_realtime_demo.py --live --port 15555

Architecture:
    SNIRF replay (or ZMQ) → RealtimePipeline → DashboardServer → browser
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser(description="Real-time fNIRS demo")
    parser.add_argument("--snirf", type=str,
                        default=str(Path.home() / "data/raw/kernel_flow/data.snirf"),
                        help="Path to SNIRF file for replay")
    parser.add_argument("--live", action="store_true",
                        help="Use ZMQ receiver instead of replay")
    parser.add_argument("--port", type=int, default=15555,
                        help="ZMQ port for live mode")
    parser.add_argument("--ws-port", type=int, default=18080,
                        help="WebSocket dashboard port")
    parser.add_argument("--mesh-nodes", type=int, default=2000,
                        help="Atlas mesh node count")
    parser.add_argument("--fast", action="store_true",
                        help="Replay at max speed (no real-time pacing)")
    args = parser.parse_args()

    print("=" * 60)
    print("dot-jax Real-Time fNIRS — Global NeuroHack Demo")
    print("=" * 60)

    # --- Load data source ---
    from dot_jax.streaming import SNIRFReplayStreamer, DashboardServer
    from dot_jax.io import read_snirf, snirf_to_dot_jax

    if args.live:
        print(f"\n[LIVE] Listening on ZMQ port {args.port}...")
        from dot_jax.streaming import ZMQFrameReceiver
        receiver = ZMQFrameReceiver(port=args.port)
        frame_source = receiver.iter_frames(timeout=5000)
        fs = 4.75  # Kernel Flow default
    else:
        print(f"\n[REPLAY] Loading {args.snirf}...")
        t0 = time.time()
        snirf = read_snirf(args.snirf)
        jd = snirf_to_dot_jax(snirf)
        fs = snirf.sampling_frequency or 4.75
        streamer = SNIRFReplayStreamer.from_arrays(
            data=np.array(jd["data"]),
            fs=fs,
            wavelength_values=np.array(jd["channel_wl"]),
            source_indices=np.array(jd["channel_src"]).astype(np.int32),
            detector_indices=np.array(jd["channel_det"]).astype(np.int32),
        )
        frame_source = streamer.iter_frames(realtime=not args.fast)
        print(f"  Loaded: {streamer.n_frames} frames x {streamer.n_channels} ch "
              f"at {fs:.2f} Hz ({time.time()-t0:.1f}s)")

    # --- Build mesh and pipeline ---
    print(f"\n[INIT] Building atlas mesh ({args.mesh_nodes} nodes)...")
    t0 = time.time()

    from dot_jax.atlas import generate_head_mesh, project_to_surface
    from dot_jax.realtime import RealtimePipeline, EpochAccumulator
    from dot_jax.hemodynamics import intensity_to_od, compute_gvtd

    mesh = generate_head_mesh(max_nodes=args.mesh_nodes)

    srcpos = jnp.array(jd["srcpos"]) if not args.live else None
    detpos = jnp.array(jd["detpos"]) if not args.live else None

    if srcpos is not None:
        src_proj = project_to_surface(mesh, np.array(srcpos))
        det_proj = project_to_surface(mesh, np.array(detpos))

        wavelengths = list(np.unique(np.array(jd["wavelengths"])).astype(float))

        print(f"  Mesh: {mesh.nn} nodes, {mesh.ne} elements")
        print(f"  Wavelengths: {wavelengths}")
        print(f"\n[INIT] Pre-computing Jacobian pseudoinverse...")

        pipe = RealtimePipeline(
            mesh, src_proj, det_proj,
            mua=0.02, musp=0.99,
            wavelengths=wavelengths,
        )
        print(f"  Pipeline ready. Channels per wv: {pipe.n_channels}")

    epoch_acc = EpochAccumulator(
        n_nodes=mesh.nn, epoch_samples=int(30 * fs), fs=fs,
    )

    print(f"  Init done in {time.time()-t0:.1f}s")

    # --- Dashboard ---
    dashboard = DashboardServer(port=args.ws_port)

    # --- Intensity baseline for OD ---
    # Collect first few frames to establish baseline
    baseline_frames = []
    baseline_n = 10

    print(f"\n[RUN] Starting pipeline...")
    print(f"  Dashboard: http://localhost:{args.ws_port}")
    print(f"  Open dashboard/index.html in browser")
    print(f"  Press Ctrl+C to stop\n")

    frame_count = 0
    prev_od = None
    gvtd_val = 0.0

    try:
        for pkt in frame_source:
            frame_count += 1

            # Collect baseline
            if frame_count <= baseline_n:
                baseline_frames.append(pkt.data.copy())
                if frame_count == baseline_n:
                    baseline = np.mean(baseline_frames, axis=0)
                    print(f"  Baseline established from {baseline_n} frames")
                continue

            # Intensity → OD
            od_raw = -np.log(pkt.data / (baseline + 1e-30))
            od_raw = np.where(np.isfinite(od_raw), od_raw, 0.0)

            # GVTD (from consecutive frames)
            if prev_od is not None:
                deriv = (od_raw - prev_od) * fs
                gvtd_val = float(np.sqrt(np.mean(deriv ** 2)))
            prev_od = od_raw.copy()

            # Split by wavelength and process
            wl_vals = np.unique(pkt.wavelength_indices)
            od_frame = {}
            for w_idx, wl_val in enumerate(sorted(wl_vals)):
                mask = pkt.wavelength_indices == wl_val
                ch_od = od_raw[mask]
                # Match to pipeline channel count
                n_need = pipe.n_channels.get(w_idx, 0)
                if len(ch_od) >= n_need and n_need > 0:
                    od_frame[w_idx] = jnp.array(ch_od[:n_need])
                else:
                    od_frame[w_idx] = jnp.zeros(n_need)

            # Reconstruct
            hbo, hbr = pipe.process_frame(od_frame)

            # Epoch accumulator
            is_event = pkt.event_type is not None
            epoch_acc.add_frame(hbo, event=is_event)

            # Format dashboard state
            epoch_result = epoch_acc.get_average()
            state = dashboard.format_state(
                hbo=np.array(hbo),
                hbr=np.array(hbr),
                gvtd=gvtd_val,
                frame_count=frame_count,
                epoch_avg=np.array(epoch_result[0]) if epoch_result else None,
            )

            # Print progress
            if frame_count % 50 == 0:
                print(f"  Frame {frame_count}: "
                      f"HbO=[{float(jnp.min(hbo)):.3e}, {float(jnp.max(hbo)):.3e}] "
                      f"GVTD={gvtd_val:.6f}")

    except KeyboardInterrupt:
        print(f"\n\nStopped at frame {frame_count}")

    print(f"\nProcessed {frame_count} frames")
    print(f"Epochs accumulated: {epoch_acc.n_epochs}")

    if args.live:
        receiver.close()


if __name__ == "__main__":
    main()
