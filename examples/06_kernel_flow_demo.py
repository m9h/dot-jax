#!/usr/bin/env python
"""Demo 6: Kernel Flow 2 TD-fNIRS — dot-jax + Holoscan Bridge.

Processes Kernel Flow 2 sample data through the dot-jax pipeline,
demonstrating interoperability between the NVIDIA Holoscan BCI
visualization pipeline and dot-jax's differentiable forward model.

Kernel Flow 2: 105 sources, 210 detectors, 6990 channels
               690/905 nm, 4.75 Hz, time-domain fNIRS

This demo shows:
  1. SNIRF I/O compatibility (same format as Holoscan BCI app)
  2. Signal processing with dot-jax hemodynamics
  3. Atlas mesh generation for Kernel Flow geometry
  4. Dual-formulation reconstruction (matching Kernel's solver)
  5. Spectral unmixing to HbO/HbR

For the Global NeuroHack, April 10-12 2026.
"""

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

SNIRF_PATH = Path(os.environ.get("DOT_JAX_DATA", Path.home() / "data/raw")) / "kernel_flow/data.snirf"

print("=" * 70)
print("dot-jax x Kernel Flow 2 — Global NeuroHack Demo")
print("=" * 70)

# ============================================================================
# 1. Load Kernel Flow SNIRF
# ============================================================================

print("\n[1] Loading Kernel Flow 2 SNIRF...")
t0 = time.time()

from dot_jax.io import read_snirf, snirf_to_dot_jax

snirf = read_snirf(str(SNIRF_PATH))
jd = snirf_to_dot_jax(snirf)

raw_data = jd["data"]
srcpos = jd["srcpos"]
detpos = jd["detpos"]
wavelengths = jd["wavelengths"]
ch_src_np = np.array(jd["channel_src"]).astype(int)
ch_det_np = np.array(jd["channel_det"]).astype(int)
ch_wl_np = np.array(jd["channel_wl"])
fs = snirf.sampling_frequency or 4.75

n_time, n_ch = raw_data.shape
n_src, n_det, n_wv = srcpos.shape[0], detpos.shape[0], len(wavelengths)

print(f"  {n_time} frames x {n_ch} channels")
print(f"  {n_src} sources, {n_det} detectors")
print(f"  Wavelengths: {np.array(wavelengths)} nm")
print(f"  Fs: {fs} Hz, Duration: {n_time/fs:.1f} s")
print(f"  Loaded in {time.time()-t0:.1f} s")

# ============================================================================
# 2. Preprocess
# ============================================================================

print("\n[2] Preprocessing...")
t0 = time.time()

from dot_jax.hemodynamics import (
    compute_channel_snr, prune_channels, intensity_to_od,
    detect_motion_artifacts, correct_motion_spline,
    bandpass_filter, downsample, compute_gvtd,
)

# Channel quality
snr = compute_channel_snr(raw_data)
good_mask = prune_channels(raw_data, snr=np.array(snr), snr_threshold=2.0)
n_good = int(np.sum(good_mask))
print(f"  Channels: {n_good}/{n_ch} pass quality ({n_ch - n_good} pruned)")

raw_good = raw_data[:, good_mask]
ch_src_g = ch_src_np[good_mask]
ch_det_g = ch_det_np[good_mask]
ch_wl_g = ch_wl_np[good_mask]

# OD → motion correction → bandpass → downsample
od = intensity_to_od(raw_good)
artifacts = detect_motion_artifacts(od, fs=fs, std_threshold=5.0)
pct = int(np.sum(artifacts)) / artifacts.size * 100
print(f"  Motion artifacts: {pct:.2f}%")
od = correct_motion_spline(od, artifacts, fs=fs)
od_filt = bandpass_filter(od, fs=fs, low=0.01, high=0.5)

# Kernel Flow is already 4.75 Hz — downsample to ~1 Hz
ds_factor = max(1, int(round(fs)))
od_ds = downsample(od_filt, factor=ds_factor)
fs_ds = fs / ds_factor

gvtd = compute_gvtd(od_ds, fs=fs_ds)
print(f"  Downsampled: {od_ds.shape[0]} frames at {fs_ds:.2f} Hz")
print(f"  GVTD: mean={float(jnp.mean(gvtd)):.6f}")
print(f"  Preprocessed in {time.time()-t0:.1f} s")

# Split by wavelength
od_per_wv = {}
src_per_wv = {}
det_per_wv = {}
for w in range(n_wv):
    mask = np.isclose(ch_wl_g, float(wavelengths[w]))
    od_per_wv[w] = od_ds[:, mask]
    src_per_wv[w] = ch_src_g[mask]
    det_per_wv[w] = ch_det_g[mask]

print(f"  Channels per wavelength: {od_per_wv[0].shape[1]} / {od_per_wv[1].shape[1]}")

# ============================================================================
# 3. Atlas mesh for Kernel Flow geometry
# ============================================================================

print("\n[3] Building atlas mesh...")
t0 = time.time()

from dot_jax.atlas import generate_head_mesh, project_to_surface

mesh = generate_head_mesh(max_nodes=3000)
src_proj = project_to_surface(mesh, np.array(srcpos))
det_proj = project_to_surface(mesh, np.array(detpos))

print(f"  Mesh: {mesh.nn} nodes, {mesh.ne} elements")
print(f"  Optodes projected to surface ({n_src} src, {n_det} det)")
print(f"  Built in {time.time()-t0:.1f} s")

# ============================================================================
# 4. Sensitivity matrix + Dual reconstruction
# ============================================================================

print("\n[4] Computing sensitivity & reconstruction...")
t0 = time.time()

from dot_jax.forward import assemble_rhs
from dot_jax.assembly import assemble_system_cw
from dot_jax.recon import solve_dual
import lineax as lx

mua_bg, musp_bg = 0.02, 0.99  # gray matter from Colin27

rhs_src = assemble_rhs(mesh, src_proj)
rhs_det = assemble_rhs(mesh, det_proj)
A = assemble_system_cw(mesh, mua_bg, musp_bg, 1.37, 1.0)

operator = lx.MatrixLinearOperator(A)
def solve_col(b):
    return lx.linear_solve(operator, b, solver=lx.LU()).value

phi_src = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_src)
phi_det = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_det)
nvol = mesh.nvol

print(f"  Green's functions: src {phi_src.shape}, det {phi_det.shape}")

# Build Rytov-normalised Jacobian + dual solve per wavelength
SD_MIN, SD_MAX = 8.0, 55.0

delta_mua = {}
for w in range(n_wv):
    src_idx = src_per_wv[w]
    det_idx = det_per_wv[w]
    sd_dist = np.sqrt(np.sum(
        (np.array(src_proj)[src_idx] - np.array(det_proj)[det_idx])**2, axis=1
    ))
    keep = (sd_dist >= SD_MIN) & (sd_dist <= SD_MAX)

    J_rows = []
    for c in np.where(keep)[0]:
        s, d = int(src_idx[c]), int(det_idx[c])
        raw_row = -phi_src[:, s] * phi_det[:, d] * nvol
        pred = rhs_det[:, d].T @ phi_src[:, s]
        J_rows.append(raw_row / jnp.maximum(jnp.abs(pred), 1e-30))

    J = jnp.stack(J_rows, axis=0)
    od_pruned = od_per_wv[w][:, keep]
    n_kept = int(np.sum(keep))
    print(f"  J[{int(wavelengths[w])} nm]: {J.shape} ({len(src_idx)-n_kept} pruned)")

    # Dual solve per timepoint (Kernel-style)
    delta_mua[w] = jnp.stack([
        solve_dual(J, od_pruned[t], reg=0.01)
        for t in range(od_pruned.shape[0])
    ], axis=0)

print(f"  Sensitivity + reconstruction in {time.time()-t0:.1f} s")

# ============================================================================
# 5. Spectral unmixing
# ============================================================================

print("\n[5] Spectral unmixing...")

from dot_jax.property import extinction
from dot_jax.hemodynamics import spectral_unmix, zscore_images
from dot_jax.mesh import smooth_on_mesh

E = extinction(list(np.array(wavelengths).astype(int)), ["hbo", "hbr"])
delta_mua_stack = jnp.stack([delta_mua[w] for w in range(n_wv)], axis=0)
delta_hbo, delta_hbr = spectral_unmix(delta_mua_stack, E)

# Smooth and z-score
delta_hbo_smooth = smooth_on_mesh(mesh, delta_hbo, fwhm=13.0)
hbo_z = zscore_images(delta_hbo_smooth)

print(f"  HbO range (raw): [{float(jnp.min(delta_hbo)):.2e}, {float(jnp.max(delta_hbo)):.2e}]")
print(f"  HbO range (smooth+z): [{float(jnp.min(hbo_z)):.2f}, {float(jnp.max(hbo_z)):.2f}]")
print(f"  Timepoints: {hbo_z.shape[0]}, Nodes: {hbo_z.shape[1]}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Kernel Flow 2 → dot-jax Pipeline")
print("=" * 70)
print(f"""
  Device:        Kernel Flow 2 TD-fNIRS
  Channels:      {n_ch} total, {n_good} after QC
  Wavelengths:   {list(np.array(wavelengths).astype(int))} nm
  Duration:      {n_time/fs:.0f} s at {fs} Hz → {od_ds.shape[0]} frames at {fs_ds:.1f} Hz
  Atlas mesh:    {mesh.nn} nodes, {mesh.ne} elements (MNI152)
  Reconstruction: Dual-formulation Tikhonov (Kernel-style)
  Output:        {hbo_z.shape[0]} x {hbo_z.shape[1]} HbO z-score images

  KEY: dot-jax provides the differentiable FEM forward model that
  generates the sensitivity matrices. The Holoscan BCI app loads
  pre-computed Jacobians; dot-jax computes them from first principles
  via the diffusion equation + adjoint method.

  With jax.grad, the entire pipeline is differentiable — enabling
  gradient-based optimization of optical properties, regularization
  parameters, and even mesh geometry.
""")

# ============================================================================
# Autodiff demonstration
# ============================================================================

print("[BONUS] Autodiff through the full pipeline...")

def signal_at_mua(mua):
    """Total detector signal as a function of absorption."""
    from dot_jax.forward import forward_cw
    r = forward_cw(mesh, mua, musp_bg,
                    src_proj[:1], det_proj[:3], 1.37, 1.0)
    return jnp.sum(r.detval)

grad_mua = jax.grad(signal_at_mua)(mua_bg)
print(f"  d(signal)/d(mua) = {float(grad_mua):.4e}")
print(f"  (automatic — no hand-derived Jacobian needed)")
print(f"\n{'=' * 70}")
print("Pipeline complete. Ready for Global NeuroHack!")
print("=" * 70)
