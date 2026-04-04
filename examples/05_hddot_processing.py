#!/usr/bin/env python
"""Tutorial 5: HD-DOT Processing Pipeline — sub-02 from ds004569.

End-to-end processing of high-density diffuse optical tomography data:

    1. Load raw SNIRF data (3684 channels, 750/850 nm, 10 Hz)
    2. Preprocess: intensity → optical density → bandpass → downsample
    3. Build atlas head mesh (MNI152) and project optodes
    4. Compute sensitivity matrices (Jacobian per wavelength)
    5. Reconstruct delta-mua per timepoint via regularised inversion
    6. Spectral unmix to HbO/HbR concentration changes
    7. Compare dot-jax HbO reconstruction to the published NIfTI

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
BANDPASS_LOW = 0.01    # Hz — remove slow drift
BANDPASS_HIGH = 0.50   # Hz — remove cardiac/respiratory
DOWNSAMPLE_FACTOR = 10  # 10 Hz → 1 Hz (match NIfTI TR)
MESH_MAX_NODES = 3000   # atlas mesh density (moderate for speed)
REG_PARAM = 0.01        # Tikhonov regularisation for reconstruction

print("=" * 70)
print("HD-DOT Processing Pipeline — dot-jax")
print("=" * 70)

# ============================================================================
# Step 1: Load raw SNIRF data
# ============================================================================

print("\n[1] Loading SNIRF data...")
t0 = time.time()

from dot_jax.io import load_bids_nirs, snirf_to_dot_jax

bids_run = load_bids_nirs(str(NIRS_DIR), task="movie")
jax_data = snirf_to_dot_jax(bids_run.snirf)

raw_data = jax_data["data"]          # (n_time, n_channels_total)
srcpos = jax_data["srcpos"]          # (n_src, 3)
detpos = jax_data["detpos"]          # (n_det, 3)
wavelengths = jax_data["wavelengths"]  # (n_wv,)
ch_src = jax_data["channel_src"]     # (n_channels_total,) 0-based
ch_det = jax_data["channel_det"]     # (n_channels_total,) 0-based
ch_wl = jax_data["channel_wl"]       # (n_channels_total,) 0-based
fs = jax_data["fs"]

n_time, n_ch = raw_data.shape
n_src = srcpos.shape[0]
n_det = detpos.shape[0]
n_wv = len(wavelengths)

print(f"  Shape: {n_time} timepoints x {n_ch} channels")
print(f"  Sources: {n_src}, Detectors: {n_det}")
print(f"  Wavelengths: {np.array(wavelengths)} nm")
print(f"  Sampling rate: {fs:.2f} Hz")
print(f"  Duration: {n_time / fs:.1f} s")
print(f"  Loaded in {time.time() - t0:.1f} s")

# ============================================================================
# Step 2: Preprocess — OD, bandpass, downsample
# ============================================================================

print("\n[2] Preprocessing...")
t0 = time.time()

from dot_jax.hemodynamics import intensity_to_od, bandpass_filter, downsample

# Intensity to optical density
od = intensity_to_od(raw_data)
print(f"  OD: shape {od.shape}, range [{float(jnp.min(od)):.4f}, {float(jnp.max(od)):.4f}]")

# Bandpass filter
od_filt = bandpass_filter(od, fs=fs, low=BANDPASS_LOW, high=BANDPASS_HIGH)
print(f"  Bandpass [{BANDPASS_LOW}–{BANDPASS_HIGH} Hz]: "
      f"range [{float(jnp.min(od_filt)):.4f}, {float(jnp.max(od_filt)):.4f}]")

# Downsample 10 Hz → 1 Hz
od_ds = downsample(od_filt, factor=DOWNSAMPLE_FACTOR)
fs_ds = fs / DOWNSAMPLE_FACTOR
print(f"  Downsampled: {od_ds.shape[0]} timepoints at {fs_ds:.2f} Hz")
print(f"  Preprocessed in {time.time() - t0:.1f} s")

# Split by wavelength (ch_wl contains wavelength values, not indices)
ch_wl_np = np.array(ch_wl)
ch_src_np = np.array(ch_src)
ch_det_np = np.array(ch_det)

od_per_wv = {}
src_per_wv = {}
det_per_wv = {}
for w in range(n_wv):
    mask = np.isclose(ch_wl_np, float(wavelengths[w]))
    od_per_wv[w] = od_ds[:, mask]
    src_per_wv[w] = ch_src_np[mask]
    det_per_wv[w] = ch_det_np[mask]

n_ch_per_wv = od_per_wv[0].shape[1]
print(f"  Channels per wavelength: {n_ch_per_wv}")

# ============================================================================
# Step 3: Build atlas mesh and project optodes
# ============================================================================

print("\n[3] Building atlas head mesh...")
t0 = time.time()

from dot_jax.atlas import generate_head_mesh, project_to_surface

mesh = generate_head_mesh(max_nodes=MESH_MAX_NODES)
print(f"  Mesh: {mesh.nn} nodes, {mesh.ne} elements, {mesh.nf} faces")
print(f"  Volume: {float(jnp.sum(mesh.evol)):.0f} mm^3")

# Project optodes to mesh surface
src_proj = project_to_surface(mesh, np.array(srcpos))
det_proj = project_to_surface(mesh, np.array(detpos))
print(f"  Sources projected: {src_proj.shape}")
print(f"  Detectors projected: {det_proj.shape}")
print(f"  Mesh built in {time.time() - t0:.1f} s")

# ============================================================================
# Step 4: Compute sensitivity matrices (Jacobian per wavelength)
# ============================================================================

print("\n[4] Computing sensitivity matrices...")
t0 = time.time()

from dot_jax.forward import assemble_rhs
from dot_jax.assembly import assemble_system_cw
from dot_jax.property import extinction

# Baseline optical properties (typical brain tissue)
mua_bg = 0.01    # 1/mm
musp_bg = 1.0    # 1/mm
n_tissue = 1.37
n_air = 1.0

# Build source/detector RHS vectors
rhs_src = assemble_rhs(mesh, src_proj)
rhs_det = assemble_rhs(mesh, det_proj)
print(f"  RHS src: {rhs_src.shape}, det: {rhs_det.shape}")

# Assemble system matrix at background properties
A = assemble_system_cw(mesh, mua_bg, musp_bg, n_tissue, n_air)
print(f"  System matrix: {A.shape}, cond ~ {float(jnp.linalg.cond(A)):.1e}")

# Solve for Green's functions (forward + adjoint)
import lineax as lx

operator = lx.MatrixLinearOperator(A)


def solve_col(b):
    return lx.linear_solve(operator, b, solver=lx.LU()).value


print("  Solving source Green's functions...")
phi_src = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_src)
print(f"  phi_src: {phi_src.shape}")

print("  Solving detector Green's functions...")
phi_det = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_det)
print(f"  phi_det: {phi_det.shape}")

# Build per-channel Jacobian with Rytov normalization:
#   J_raw[ch, n] = -phi_src[n, s] * phi_det[n, d] * nvol[n]
#   predicted[ch] = rhs_det[:, d].T @ phi_src[:, s]
#   J_norm[ch, n] = J_raw[ch, n] / predicted[ch]
# This converts absolute sensitivity to log-ratio sensitivity (matching OD data).
nvol = mesh.nvol

# Also prune channels by source-detector distance (keep 13–50 mm)
SD_MIN, SD_MAX = 10.0, 55.0  # mm

J_per_wv = {}
od_per_wv_pruned = {}
keep_masks = {}

for w in range(n_wv):
    src_idx = src_per_wv[w]
    det_idx = det_per_wv[w]
    n_ch_w = len(src_idx)

    # Compute SD distances and prune
    sd_dist = np.sqrt(np.sum(
        (np.array(src_proj)[src_idx] - np.array(det_proj)[det_idx]) ** 2, axis=1
    ))
    keep = (sd_dist >= SD_MIN) & (sd_dist <= SD_MAX)
    keep_masks[w] = keep

    src_keep = src_idx[keep]
    det_keep = det_idx[keep]
    n_keep = int(np.sum(keep))

    J_rows = []
    for c in range(n_keep):
        s = int(src_keep[c])
        d = int(det_keep[c])
        raw_row = -phi_src[:, s] * phi_det[:, d] * nvol
        predicted = rhs_det[:, d].T @ phi_src[:, s]
        # Rytov normalisation: divide by predicted measurement
        norm_row = raw_row / jnp.maximum(jnp.abs(predicted), 1e-30)
        J_rows.append(norm_row)

    J_per_wv[w] = jnp.stack(J_rows, axis=0)
    od_per_wv_pruned[w] = od_per_wv[w][:, keep]
    print(f"  J[{int(wavelengths[w])} nm]: {J_per_wv[w].shape} "
          f"({n_ch_w - n_keep} channels pruned by SD distance)")

print(f"  Sensitivity computed in {time.time() - t0:.1f} s")

# ============================================================================
# Step 5: Reconstruct delta_mua per timepoint
# ============================================================================

print("\n[5] Reconstructing absorption images...")
t0 = time.time()

# Precompute regularised pseudo-inverse for each wavelength
# (J^T J + lambda * I)^{-1} J^T  (standard Tikhonov)
Jinv_per_wv = {}
for w in range(n_wv):
    J = J_per_wv[w]
    JtJ = J.T @ J
    # Scale regularization by trace for numerical stability
    reg = REG_PARAM * jnp.trace(JtJ) / J.shape[1] * jnp.eye(J.shape[1])
    Jinv_per_wv[w] = jnp.linalg.solve(JtJ + reg, J.T)  # (nn, n_ch_w)
    print(f"  Jinv[{int(wavelengths[w])} nm]: {Jinv_per_wv[w].shape}")

# Reconstruct: delta_mua[w, t, :] = Jinv @ od_pruned[t, :]
n_time_ds = od_ds.shape[0]
delta_mua = {}
for w in range(n_wv):
    delta_mua[w] = (Jinv_per_wv[w] @ od_per_wv_pruned[w].T).T  # (n_time, nn)
    print(f"  delta_mua[{int(wavelengths[w])} nm]: {delta_mua[w].shape}, "
          f"range [{float(jnp.min(delta_mua[w])):.4e}, {float(jnp.max(delta_mua[w])):.4e}]")

print(f"  Reconstruction done in {time.time() - t0:.1f} s")

# ============================================================================
# Step 6: Spectral unmix to HbO/HbR
# ============================================================================

print("\n[6] Spectral unmixing...")

E = extinction(list(np.array(wavelengths)), ["hbo", "hbr"])
print(f"  Extinction matrix:\n    {np.array(E)}")

from dot_jax.hemodynamics import spectral_unmix

# Stack wavelengths: (n_wv, n_time, nn)
delta_mua_stack = jnp.stack([delta_mua[w] for w in range(n_wv)], axis=0)
delta_hbo, delta_hbr = spectral_unmix(delta_mua_stack, E)

print(f"  delta_HbO: {delta_hbo.shape}, range [{float(jnp.min(delta_hbo)):.4e}, {float(jnp.max(delta_hbo)):.4e}]")
print(f"  delta_HbR: {delta_hbr.shape}, range [{float(jnp.min(delta_hbr)):.4e}, {float(jnp.max(delta_hbr)):.4e}]")

# ============================================================================
# Step 7: Compare to published NIfTI
# ============================================================================

print("\n[7] Comparing to published NIfTI reconstruction...")

try:
    import nibabel as nib
    from scipy.spatial import cKDTree

    nii = nib.load(str(NIFTI_FILE))
    nii_data = np.array(nii.get_fdata())
    nii_affine = nii.affine

    print(f"  NIfTI shape: {nii_data.shape}")
    print(f"  NIfTI voxel size: {nii.header.get_zooms()[:3]} mm")
    print(f"  NIfTI time points: {nii_data.shape[-1]}")

    # Map mesh nodes to NIfTI voxel space
    node_mm = np.array(mesh.node)
    # Inverse affine: mm → voxel
    inv_affine = np.linalg.inv(nii_affine)
    node_vox = (inv_affine[:3, :3] @ node_mm.T + inv_affine[:3, 3:4]).T

    # Extract NIfTI time series at each mesh node (nearest-neighbor)
    node_vox_int = np.round(node_vox).astype(int)
    valid = np.all(
        (node_vox_int >= 0) & (node_vox_int < np.array(nii_data.shape[:3])),
        axis=1,
    )

    n_valid = np.sum(valid)
    print(f"  Mesh nodes inside NIfTI volume: {n_valid}/{mesh.nn}")

    # Trim time axes to common length
    # NIfTI has movie onset at t=30s, our data starts at t=0
    # The NIfTI description says "trimmed to movie onset" so time 0 = movie start
    # Our preprocessing: raw starts at 0, movie at 30s, we downsample to 1Hz
    # Trim to movie onset (30s into recording = sample 30 at 1Hz)
    movie_onset_sample = int(30 * fs_ds)
    hbo_movie = np.array(delta_hbo[movie_onset_sample:])
    n_common = min(hbo_movie.shape[0], nii_data.shape[-1])

    hbo_trimmed = hbo_movie[:n_common]  # (n_common, nn)
    nii_trimmed = nii_data[:, :, :, :n_common]  # (x, y, z, n_common)

    # Extract NIfTI values at valid node locations
    nii_at_nodes = np.zeros((n_common, mesh.nn))
    for i in range(mesh.nn):
        if valid[i]:
            vx, vy, vz = node_vox_int[i]
            nii_at_nodes[:, i] = nii_trimmed[vx, vy, vz, :]

    # Diagnostics: check NIfTI non-zero coverage at mesh nodes
    nii_nonzero = np.sum(np.abs(nii_at_nodes) > 1e-10, axis=0) > 0
    n_nii_active = np.sum(nii_nonzero & valid)
    print(f"  NIfTI non-zero at mesh nodes: {n_nii_active}/{n_valid}")

    # Use only nodes where NIfTI has signal AND inside volume
    active = valid & nii_nonzero

    # Global mean time series comparison (sanity check)
    global_dotjax = np.mean(np.array(hbo_trimmed[:, active]), axis=1)
    global_nifti = np.mean(nii_at_nodes[:, active], axis=1)

    if np.std(global_dotjax) > 0 and np.std(global_nifti) > 0:
        global_corr = np.corrcoef(global_dotjax, global_nifti)[0, 1]
    else:
        global_corr = 0.0

    print(f"\n  === Global Mean Signal ===")
    print(f"  dot-jax HbO range: [{global_dotjax.min():.4f}, {global_dotjax.max():.4f}]")
    print(f"  NIfTI HbO range:   [{global_nifti.min():.4f}, {global_nifti.max():.4f}]")
    print(f"  Global temporal correlation: {global_corr:.4f}")

    # Spatial correlation at each timepoint
    corr_per_time = []
    for t in range(n_common):
        x = np.array(hbo_trimmed[t, active])
        y = nii_at_nodes[t, active]
        if np.std(x) > 0 and np.std(y) > 0:
            c = np.corrcoef(x, y)[0, 1]
            corr_per_time.append(c)

    corr_per_time = np.array(corr_per_time)
    valid_corr = corr_per_time[np.isfinite(corr_per_time)]

    print(f"\n  === Spatial Correlation (dot-jax HbO vs NIfTI HbO) ===")
    print(f"  Active nodes: {int(np.sum(active))}")
    print(f"  Time points compared: {len(valid_corr)}")
    print(f"  Mean correlation: {np.mean(valid_corr):.4f}")
    print(f"  Median correlation: {np.median(valid_corr):.4f}")
    print(f"  Range: [{np.min(valid_corr):.4f}, {np.max(valid_corr):.4f}]")

    # Temporal correlation at each node
    temp_corr = []
    node_ids = []
    for i in range(mesh.nn):
        if active[i]:
            x = np.array(hbo_trimmed[:, i])
            y = nii_at_nodes[:, i]
            if np.std(x) > 0 and np.std(y) > 0:
                c = np.corrcoef(x, y)[0, 1]
                temp_corr.append(c)
                node_ids.append(i)

    temp_corr = np.array(temp_corr)
    valid_temp = temp_corr[np.isfinite(temp_corr)]

    print(f"\n  === Temporal Correlation (per node, dot-jax vs NIfTI) ===")
    print(f"  Nodes compared: {len(valid_temp)}")
    if len(valid_temp) > 0:
        print(f"  Mean correlation: {np.mean(valid_temp):.4f}")
        print(f"  Median correlation: {np.median(valid_temp):.4f}")
        print(f"  Range: [{np.min(valid_temp):.4f}, {np.max(valid_temp):.4f}]")
        print(f"  Nodes with r > 0.3: {np.sum(valid_temp > 0.3)}/{len(valid_temp)}")
        print(f"  Nodes with r > 0.5: {np.sum(valid_temp > 0.5)}/{len(valid_temp)}")
        # Top 5 nodes by correlation
        top5 = np.argsort(valid_temp)[-5:][::-1]
        print(f"  Top 5 nodes: {[f'r={valid_temp[i]:.3f}' for i in top5]}")

except ImportError as e:
    print(f"  Skipping NIfTI comparison (missing {e.name})")
except FileNotFoundError:
    print(f"  NIfTI file not found: {NIFTI_FILE}")

print("\n" + "=" * 70)
print("Pipeline complete.")
print("=" * 70)
