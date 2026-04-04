#!/usr/bin/env python
"""Tutorial 5: HD-DOT Processing Pipeline — sub-02 from ds004569.

End-to-end processing of high-density diffuse optical tomography data
with full signal processing, artifact correction, and normalization:

    1. Load raw SNIRF data (3684 channels, 750/850 nm, 10 Hz)
    2. Channel quality pruning (SNR, CV, saturation)
    3. Intensity → optical density
    4. Motion artifact detection and spline correction
    5. Short-separation regression (remove superficial signal)
    6. Bandpass filter and downsample
    7. Build atlas head mesh (MNI152) and project optodes
    8. Compute Rytov-normalised sensitivity matrices
    9. GCV-optimised regularisation and image reconstruction
   10. Spectral unmix to HbO/HbR
   11. Spatial smoothing (13 mm FWHM to match NIfTI)
   12. Z-score and compare to published NIfTI tomography

Data: OpenNeuro ds004569 — Sherafati et al. (2025), Sci. Data.
"""

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT = Path(os.environ.get("DOT_JAX_DATA", Path.home() / "data/raw"))
DS = DATA_ROOT / "ds004569"
SUBJECT = "sub-02"
SESSION = "ses-01"
NIRS_DIR = DS / SUBJECT / SESSION / "nirs"
NIFTI_FILE = DS / SUBJECT / SESSION / "func" / f"{SUBJECT}_{SESSION}_task-movie_bold.nii.gz"

# Processing parameters
SNR_THRESHOLD = 2.0
MOTION_STD_THRESHOLD = 5.0
SHORT_SD_MAX = 10.0       # mm — channels shorter than this are "short-separation"
BANDPASS_LOW = 0.01       # Hz
BANDPASS_HIGH = 0.50      # Hz
DOWNSAMPLE_FACTOR = 10    # 10 Hz → 1 Hz
MESH_MAX_NODES = 3000
SD_KEEP_MIN = 10.0        # mm — reconstruction channel range
SD_KEEP_MAX = 55.0        # mm
SMOOTH_FWHM = 13.0        # mm — match NIfTI's 13mm kernel

print("=" * 70)
print("HD-DOT Processing Pipeline — dot-jax (v2: full signal processing)")
print("=" * 70)

# ============================================================================
# Step 1: Load raw SNIRF data
# ============================================================================

print("\n[1] Loading SNIRF data...")
t0 = time.time()

from dot_jax.io import load_bids_nirs, snirf_to_dot_jax

bids_run = load_bids_nirs(str(NIRS_DIR), task="movie")
jax_data = snirf_to_dot_jax(bids_run.snirf)

raw_data = jax_data["data"]
srcpos = jax_data["srcpos"]
detpos = jax_data["detpos"]
wavelengths = jax_data["wavelengths"]
ch_src = jax_data["channel_src"]
ch_det = jax_data["channel_det"]
ch_wl = jax_data["channel_wl"]
fs = jax_data["fs"]

n_time, n_ch = raw_data.shape
n_src, n_det, n_wv = srcpos.shape[0], detpos.shape[0], len(wavelengths)

print(f"  {n_time} timepoints x {n_ch} channels, {n_src} src, {n_det} det")
print(f"  Wavelengths: {np.array(wavelengths)} nm, fs = {fs:.2f} Hz")
print(f"  Loaded in {time.time() - t0:.1f} s")

# ============================================================================
# Step 2: Channel quality pruning
# ============================================================================

print("\n[2] Channel quality pruning...")
t0 = time.time()

from dot_jax.hemodynamics import (
    compute_channel_snr, prune_channels, intensity_to_od,
    detect_motion_artifacts, correct_motion_spline,
    identify_short_channels, regress_short_channels,
    bandpass_filter, downsample, spectral_unmix,
    normalize_od, zscore_images, compute_gvtd,
)

snr = compute_channel_snr(raw_data)
good_mask = prune_channels(raw_data, snr=np.array(snr), snr_threshold=SNR_THRESHOLD)
n_good = int(np.sum(good_mask))
n_bad = n_ch - n_good

print(f"  SNR range: [{float(jnp.min(snr)):.1f}, {float(jnp.max(snr)):.1f}]")
print(f"  Good channels: {n_good}/{n_ch} ({n_bad} pruned)")

# Apply mask — keep only good channels
raw_good = raw_data[:, good_mask]
ch_src_good = np.array(ch_src)[good_mask]
ch_det_good = np.array(ch_det)[good_mask]
ch_wl_good = np.array(ch_wl)[good_mask]

print(f"  Pruning done in {time.time() - t0:.1f} s")

# ============================================================================
# Step 3: Intensity to optical density
# ============================================================================

print("\n[3] Intensity → optical density...")
t0 = time.time()

od = intensity_to_od(raw_good)
print(f"  OD: {od.shape}, range [{float(jnp.min(od)):.4f}, {float(jnp.max(od)):.4f}]")

# ============================================================================
# Step 4: Motion artifact correction
# ============================================================================

print("\n[4] Motion artifact detection & correction...")

artifact_mask = detect_motion_artifacts(od, fs=fs, std_threshold=MOTION_STD_THRESHOLD)
n_artifacts = int(np.sum(artifact_mask))
pct_artifact = n_artifacts / artifact_mask.size * 100
print(f"  Artifacts flagged: {n_artifacts}/{artifact_mask.size} samples ({pct_artifact:.2f}%)")

od_corrected = correct_motion_spline(od, artifact_mask, fs=fs)
print(f"  Spline correction applied")

# ============================================================================
# Step 5: Short-separation regression
# ============================================================================

print("\n[5] Short-separation regression...")

short_mask = identify_short_channels(
    srcpos, detpos, ch_src_good, ch_det_good, max_distance=SHORT_SD_MAX,
)
n_short = int(np.sum(short_mask))
print(f"  Short channels (SD < {SHORT_SD_MAX} mm): {n_short}/{len(short_mask)}")

if n_short > 0:
    od_regressed = regress_short_channels(od_corrected, short_mask)
    print(f"  Superficial signal regressed from {int(np.sum(~short_mask))} long channels")
else:
    od_regressed = od_corrected
    print(f"  No short channels found — skipping regression")

# ============================================================================
# Step 6: Bandpass filter and downsample
# ============================================================================

print("\n[6] Bandpass filter & downsample...")
t0 = time.time()

od_filt = bandpass_filter(od_regressed, fs=fs, low=BANDPASS_LOW, high=BANDPASS_HIGH)
od_ds = downsample(od_filt, factor=DOWNSAMPLE_FACTOR)
fs_ds = fs / DOWNSAMPLE_FACTOR

# GVTD quality metric
gvtd = compute_gvtd(od_ds, fs=fs_ds)
print(f"  Filtered & downsampled: {od_ds.shape[0]} timepoints at {fs_ds:.2f} Hz")
print(f"  GVTD: mean = {float(jnp.mean(gvtd)):.6f}, max = {float(jnp.max(gvtd)):.6f}")

# Split by wavelength
od_per_wv = {}
src_per_wv = {}
det_per_wv = {}
for w in range(n_wv):
    mask = np.isclose(ch_wl_good, float(wavelengths[w]))
    od_per_wv[w] = od_ds[:, mask]
    src_per_wv[w] = ch_src_good[mask]
    det_per_wv[w] = ch_det_good[mask]

print(f"  Channels per wavelength: {od_per_wv[0].shape[1]}")
print(f"  Preprocessed in {time.time() - t0:.1f} s")

# ============================================================================
# Step 7: Build atlas mesh and project optodes
# ============================================================================

print("\n[7] Building atlas head mesh...")
t0 = time.time()

from dot_jax.atlas import generate_head_mesh, project_to_surface

mesh = generate_head_mesh(max_nodes=MESH_MAX_NODES)
src_proj = project_to_surface(mesh, np.array(srcpos))
det_proj = project_to_surface(mesh, np.array(detpos))

print(f"  Mesh: {mesh.nn} nodes, {mesh.ne} elements")
print(f"  Optodes projected to surface")
print(f"  Built in {time.time() - t0:.1f} s")

# ============================================================================
# Step 8: Compute Rytov-normalised sensitivity matrices
# ============================================================================

print("\n[8] Computing sensitivity matrices...")
t0 = time.time()

from dot_jax.forward import assemble_rhs
from dot_jax.assembly import assemble_system_cw
from dot_jax.property import extinction
import lineax as lx

mua_bg, musp_bg = 0.01, 1.0
n_tissue, n_air = 1.37, 1.0

rhs_src = assemble_rhs(mesh, src_proj)
rhs_det = assemble_rhs(mesh, det_proj)
A = assemble_system_cw(mesh, mua_bg, musp_bg, n_tissue, n_air)
operator = lx.MatrixLinearOperator(A)

def solve_col(b):
    return lx.linear_solve(operator, b, solver=lx.LU()).value

phi_src = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_src)
phi_det = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_det)

nvol = mesh.nvol

# Build Jacobian per wavelength: Rytov-normalised, SD-distance pruned
J_per_wv = {}
od_per_wv_pruned = {}

for w in range(n_wv):
    src_idx = src_per_wv[w]
    det_idx = det_per_wv[w]
    n_ch_w = len(src_idx)

    sd_dist = np.sqrt(np.sum(
        (np.array(src_proj)[src_idx] - np.array(det_proj)[det_idx]) ** 2, axis=1
    ))
    keep = (sd_dist >= SD_KEEP_MIN) & (sd_dist <= SD_KEEP_MAX)

    src_keep = src_idx[keep]
    det_keep = det_idx[keep]

    J_rows = []
    for c in range(int(np.sum(keep))):
        s, d = int(src_keep[c]), int(det_keep[c])
        raw_row = -phi_src[:, s] * phi_det[:, d] * nvol
        predicted = rhs_det[:, d].T @ phi_src[:, s]
        J_rows.append(raw_row / jnp.maximum(jnp.abs(predicted), 1e-30))

    J_per_wv[w] = jnp.stack(J_rows, axis=0)
    od_per_wv_pruned[w] = od_per_wv[w][:, keep]
    print(f"  J[{int(wavelengths[w])} nm]: {J_per_wv[w].shape} "
          f"({n_ch_w - int(np.sum(keep))} pruned)")

print(f"  Sensitivity computed in {time.time() - t0:.1f} s")

# ============================================================================
# Step 9: GCV-optimised regularisation and reconstruction
# ============================================================================

print("\n[9] Reconstructing with GCV-optimised regularisation...")
t0 = time.time()

from dot_jax.recon import select_lambda_gcv

# Use a representative timepoint for GCV optimisation
mid_t = od_per_wv_pruned[0].shape[0] // 2

Jinv_per_wv = {}
for w in range(n_wv):
    J = J_per_wv[w]
    data_sample = od_per_wv_pruned[w][mid_t, :]

    lambda_opt = select_lambda_gcv(J, data_sample)
    print(f"  lambda_GCV[{int(wavelengths[w])} nm] = {lambda_opt:.4e}")

    JtJ = J.T @ J
    reg = lambda_opt * jnp.eye(J.shape[1])
    Jinv_per_wv[w] = jnp.linalg.solve(JtJ + reg, J.T)

delta_mua = {}
for w in range(n_wv):
    delta_mua[w] = (Jinv_per_wv[w] @ od_per_wv_pruned[w].T).T
    print(f"  delta_mua[{int(wavelengths[w])} nm]: range "
          f"[{float(jnp.min(delta_mua[w])):.4e}, {float(jnp.max(delta_mua[w])):.4e}]")

print(f"  Reconstruction done in {time.time() - t0:.1f} s")

# ============================================================================
# Step 10: Spectral unmixing
# ============================================================================

print("\n[10] Spectral unmixing...")

E = extinction(list(np.array(wavelengths)), ["hbo", "hbr"])
delta_mua_stack = jnp.stack([delta_mua[w] for w in range(n_wv)], axis=0)
delta_hbo, delta_hbr = spectral_unmix(delta_mua_stack, E)

print(f"  delta_HbO range: [{float(jnp.min(delta_hbo)):.4e}, {float(jnp.max(delta_hbo)):.4e}]")
print(f"  delta_HbR range: [{float(jnp.min(delta_hbr)):.4e}, {float(jnp.max(delta_hbr)):.4e}]")

# ============================================================================
# Step 11: Spatial smoothing (13 mm FWHM)
# ============================================================================

print(f"\n[11] Spatial smoothing (FWHM = {SMOOTH_FWHM} mm)...")
t0 = time.time()

from dot_jax.mesh import smooth_on_mesh

delta_hbo_smooth = smooth_on_mesh(mesh, delta_hbo, fwhm=SMOOTH_FWHM)
delta_hbr_smooth = smooth_on_mesh(mesh, delta_hbr, fwhm=SMOOTH_FWHM)

print(f"  Smoothed HbO range: [{float(jnp.min(delta_hbo_smooth)):.4e}, "
      f"{float(jnp.max(delta_hbo_smooth)):.4e}]")
print(f"  Smoothed in {time.time() - t0:.1f} s")

# ============================================================================
# Step 12: Z-score and compare to NIfTI
# ============================================================================

print("\n[12] Z-scoring and comparing to NIfTI...")

# Z-score the dot-jax images
hbo_z = zscore_images(delta_hbo_smooth)
print(f"  dot-jax HbO z-scored: mean={float(jnp.mean(hbo_z)):.4f}, "
      f"std={float(jnp.std(hbo_z)):.4f}")

try:
    import nibabel as nib

    nii = nib.load(str(NIFTI_FILE))
    nii_data = np.array(nii.get_fdata())
    nii_affine = nii.affine

    print(f"  NIfTI: {nii_data.shape}, {nii.header.get_zooms()[:3]} mm voxels")

    # Map mesh nodes to NIfTI voxels
    node_mm = np.array(mesh.node)
    inv_affine = np.linalg.inv(nii_affine)
    node_vox = (inv_affine[:3, :3] @ node_mm.T + inv_affine[:3, 3:4]).T
    node_vox_int = np.round(node_vox).astype(int)
    valid = np.all(
        (node_vox_int >= 0) & (node_vox_int < np.array(nii_data.shape[:3])),
        axis=1,
    )

    # Trim to movie onset and common length
    movie_onset_sample = int(30 * fs_ds)
    hbo_movie = np.array(hbo_z[movie_onset_sample:])
    n_common = min(hbo_movie.shape[0], nii_data.shape[-1])
    hbo_trimmed = hbo_movie[:n_common]
    nii_trimmed = nii_data[:, :, :, :n_common]

    # Extract NIfTI at mesh nodes
    nii_at_nodes = np.zeros((n_common, mesh.nn))
    for i in range(mesh.nn):
        if valid[i]:
            vx, vy, vz = node_vox_int[i]
            nii_at_nodes[:, i] = nii_trimmed[vx, vy, vz, :]

    # Z-score the NIfTI time series too
    nii_mu = np.mean(nii_at_nodes, axis=0, keepdims=True)
    nii_std = np.std(nii_at_nodes, axis=0, keepdims=True)
    nii_z = np.where(nii_std > 1e-15, (nii_at_nodes - nii_mu) / nii_std, 0.0)

    # Active nodes: inside volume AND NIfTI non-zero
    nii_nonzero = np.any(np.abs(nii_at_nodes) > 1e-10, axis=0)
    active = valid & nii_nonzero
    n_active = int(np.sum(active))
    print(f"  Active nodes (NIfTI non-zero & in volume): {n_active}/{mesh.nn}")

    # --- Global mean comparison ---
    global_dotjax = np.mean(hbo_trimmed[:, active], axis=1)
    global_nifti = np.mean(nii_z[:, active], axis=1)
    if np.std(global_dotjax) > 0 and np.std(global_nifti) > 0:
        global_corr = np.corrcoef(global_dotjax, global_nifti)[0, 1]
    else:
        global_corr = 0.0

    print(f"\n  === Global Mean Temporal Correlation ===")
    print(f"  r = {global_corr:.4f}")

    # --- Spatial correlation per timepoint ---
    corr_spatial = []
    for t in range(n_common):
        x, y = hbo_trimmed[t, active], nii_z[t, active]
        if np.std(x) > 0 and np.std(y) > 0:
            corr_spatial.append(np.corrcoef(x, y)[0, 1])
    corr_spatial = np.array(corr_spatial)
    valid_s = corr_spatial[np.isfinite(corr_spatial)]

    print(f"\n  === Spatial Correlation (per timepoint) ===")
    print(f"  Time points: {len(valid_s)}")
    print(f"  Mean r = {np.mean(valid_s):.4f}, Median r = {np.median(valid_s):.4f}")
    print(f"  Range: [{np.min(valid_s):.4f}, {np.max(valid_s):.4f}]")

    # --- Temporal correlation per node ---
    corr_temporal = []
    for i in range(mesh.nn):
        if active[i]:
            x, y = hbo_trimmed[:, i], nii_z[:, i]
            if np.std(x) > 0 and np.std(y) > 0:
                corr_temporal.append(np.corrcoef(x, y)[0, 1])
    corr_temporal = np.array(corr_temporal)
    valid_t = corr_temporal[np.isfinite(corr_temporal)]

    print(f"\n  === Temporal Correlation (per node) ===")
    print(f"  Nodes: {len(valid_t)}")
    if len(valid_t) > 0:
        print(f"  Mean r = {np.mean(valid_t):.4f}, Median r = {np.median(valid_t):.4f}")
        print(f"  Range: [{np.min(valid_t):.4f}, {np.max(valid_t):.4f}]")
        print(f"  Nodes r > 0.1: {np.sum(valid_t > 0.1)}/{len(valid_t)}")
        print(f"  Nodes r > 0.3: {np.sum(valid_t > 0.3)}/{len(valid_t)}")
        print(f"  Nodes r > 0.5: {np.sum(valid_t > 0.5)}/{len(valid_t)}")
        top5 = np.argsort(valid_t)[-5:][::-1]
        print(f"  Top 5: {[f'r={valid_t[i]:.3f}' for i in top5]}")

except ImportError as e:
    print(f"  Skipping NIfTI comparison (missing {e.name})")
except FileNotFoundError:
    print(f"  NIfTI file not found: {NIFTI_FILE}")

print("\n" + "=" * 70)
print("Pipeline complete.")
print("=" * 70)
