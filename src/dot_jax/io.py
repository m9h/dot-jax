"""SNIRF and BIDS-fNIRS I/O for dot-jax.

Reads SNIRF files (HDF5) and BIDS-fNIRS directory layouts into
JAX-compatible arrays for the forward/inverse pipeline. Uses h5py
directly — no external SNIRF library required.

SNIRF spec: https://github.com/fNIRS/snirf
BIDS-fNIRS: https://bids-specification.readthedocs.io/en/stable/
             modality-specific-files/near-infrared-spectroscopy.html

Functions:
    read_snirf: Read a .snirf file into a SnirfData container
    load_bids_nirs: Load a BIDS-fNIRS subject/session directory
    snirf_to_dot_jax: Extract arrays needed for forward_cw / reconstruct_mua
"""

import csv
import json
from pathlib import Path
from typing import NamedTuple, Optional

import h5py
import jax.numpy as jnp
import numpy as np


class MeasurementList(NamedTuple):
    """Per-channel metadata from SNIRF measurementList."""
    source_index: np.ndarray    # (n_chan,) 1-based source indices
    detector_index: np.ndarray  # (n_chan,) 1-based detector indices
    wavelength_index: np.ndarray  # (n_chan,) 1-based wavelength indices
    data_type: np.ndarray       # (n_chan,) SNIRF data type codes


class SnirfData(NamedTuple):
    """Data extracted from a SNIRF file."""
    data: np.ndarray               # (n_time, n_chan) measurement time series
    source_pos: np.ndarray         # (n_src, 3) source 3D positions [mm]
    detector_pos: np.ndarray       # (n_det, 3) detector 3D positions [mm]
    wavelengths: np.ndarray        # (n_wl,) wavelength list [nm]
    measurement_list: MeasurementList
    sampling_frequency: Optional[float]  # Hz, if available


class BidsNirsRun(NamedTuple):
    """Data from a single BIDS-fNIRS run."""
    snirf: SnirfData
    task_name: Optional[str]
    events: Optional[np.ndarray]         # (n_events, 3) onset/duration/trial_type
    optodes_tsv: Optional[dict]          # parsed optodes.tsv
    channels_tsv: Optional[dict]         # parsed channels.tsv
    coordsystem: Optional[dict]          # parsed coordsystem.json
    metadata: Optional[dict]             # parsed _nirs.json sidecar


# =============================================================================
# SNIRF reader (h5py)
# =============================================================================


def read_snirf(path, nirs_idx=0, data_idx=0):
    """Read a .snirf file into a SnirfData container.

    Parameters
    ----------
    path : str or Path
        Path to .snirf file.
    nirs_idx : int
        Which /nirs group to read (0-based, maps to /nirs1, /nirs2, ...).
    data_idx : int
        Which data block to read (0-based, maps to /data1, /data2, ...).

    Returns
    -------
    SnirfData
    """
    path = Path(path)

    with h5py.File(path, "r") as f:
        # SNIRF uses 1-based group names: /nirs1, /nirs2, ...
        # but some files use /nirs/nirs1 or /nirs/data1 patterns
        nirs_key = _find_group(f, "nirs", nirs_idx)
        nirs = f[nirs_key]

        data_key = _find_group(nirs, "data", data_idx)
        data_grp = nirs[data_key]

        # Time series: (n_time, n_chan)
        data = data_grp["dataTimeSeries"][:]

        # Measurement list
        ml = _read_measurement_list(data_grp)

        # Probe geometry
        probe = nirs["probe"]
        source_pos = _read_positions(probe, "sourcePos")
        detector_pos = _read_positions(probe, "detectorPos")
        wavelengths = probe["wavelengths"][:]

        # Sampling frequency (try several common locations)
        fs = None
        if "metaDataTags" in nirs:
            meta = nirs["metaDataTags"]
            for fs_key in ("SamplingFrequency", "framerate", "Framerate"):
                if fs_key in meta:
                    fs = float(meta[fs_key][()])
                    break

    return SnirfData(
        data=data,
        source_pos=source_pos,
        detector_pos=detector_pos,
        wavelengths=wavelengths,
        measurement_list=ml,
        sampling_frequency=fs,
    )


def _find_group(parent, prefix, idx):
    """Find HDF5 group by prefix and 0-based index.

    Tries common SNIRF naming patterns:
    - '{prefix}{idx+1}' (e.g., 'nirs1', 'data1')
    - '{prefix}' if only one group exists
    - '{prefix}/{prefix}{idx+1}'
    """
    # Most common: prefix + 1-based index
    key1 = f"{prefix}{idx + 1}"
    if key1 in parent:
        return key1

    # Single group without index
    if prefix in parent:
        return prefix

    # Try as subpath
    key2 = f"{prefix}/{prefix}{idx + 1}"
    if key2 in parent:
        return key2

    # Fallback: find first matching key
    for key in parent.keys():
        if key.startswith(prefix):
            return key

    raise KeyError(f"Cannot find group '{prefix}' (index {idx}) in {parent.name}")


def _read_positions(probe, prefix):
    """Read source/detector positions, preferring 3D over 2D."""
    key_3d = f"{prefix}3D"
    key_2d = f"{prefix}2D"

    if key_3d in probe:
        return probe[key_3d][:]
    elif key_2d in probe:
        pos_2d = probe[key_2d][:]
        # Pad 2D positions with z=0
        return np.column_stack([pos_2d, np.zeros(pos_2d.shape[0])])
    else:
        raise KeyError(f"Neither {key_3d} nor {key_2d} found in probe group")


def _read_measurement_list(data_grp):
    """Read measurement list from SNIRF data group.

    Handles indexed groups (measurementList1, ..., measurementListN).
    For large channel counts (>100), reads keys in sorted batch to
    avoid O(N*log(N)) overhead from repeated key lookups.
    """
    # Count measurement list groups
    ml_keys = [k for k in data_grp.keys() if k.startswith("measurementList")]
    n_chan = len(ml_keys)

    if n_chan == 0:
        return MeasurementList(
            source_index=np.array([], dtype=np.int32),
            detector_index=np.array([], dtype=np.int32),
            wavelength_index=np.array([], dtype=np.int32),
            data_type=np.array([], dtype=np.int32),
        )

    src_idx = np.zeros(n_chan, dtype=np.int32)
    det_idx = np.zeros(n_chan, dtype=np.int32)
    wl_idx = np.zeros(n_chan, dtype=np.int32)
    dtype = np.ones(n_chan, dtype=np.int32)

    # Read sequentially by index (1-based per SNIRF spec)
    for i in range(n_chan):
        key = f"measurementList{i + 1}"
        ml = data_grp[key]
        src_idx[i] = int(ml["sourceIndex"][()])
        det_idx[i] = int(ml["detectorIndex"][()])
        wl_idx[i] = int(ml["wavelengthIndex"][()])
        if "dataType" in ml:
            dtype[i] = int(ml["dataType"][()])

    return MeasurementList(
        source_index=src_idx,
        detector_index=det_idx,
        wavelength_index=wl_idx,
        data_type=dtype,
    )


def _ml_sort_key(key):
    """Sort measurementList keys numerically."""
    digits = "".join(c for c in key if c.isdigit())
    return int(digits) if digits else 0


# =============================================================================
# BIDS-fNIRS loader
# =============================================================================


def load_bids_nirs(nirs_dir, task=None, run=None):
    """Load a BIDS-fNIRS directory (subject/session/nirs/).

    Parameters
    ----------
    nirs_dir : str or Path
        Path to the nirs/ directory (e.g., sub-01/ses-01/nirs/).
    task : str, optional
        Task name filter. If None, loads first .snirf found.
    run : int, optional
        Run number filter.

    Returns
    -------
    BidsNirsRun
    """
    nirs_dir = Path(nirs_dir)

    # Find the .snirf file
    snirf_files = sorted(nirs_dir.glob("*.snirf"))
    if task:
        snirf_files = [f for f in snirf_files if f"task-{task}" in f.name]
    if run is not None:
        snirf_files = [f for f in snirf_files if f"run-{run:02d}" in f.name]
    if not snirf_files:
        raise FileNotFoundError(
            f"No .snirf files found in {nirs_dir}"
            + (f" for task={task}" if task else "")
        )

    snirf_path = snirf_files[0]
    stem = snirf_path.stem  # e.g., sub-01_ses-01_task-movie

    # Read SNIRF
    snirf = read_snirf(snirf_path)

    # Read sidecar JSON
    json_path = nirs_dir / f"{stem}_nirs.json"
    metadata = _read_json(json_path)

    # Override sampling frequency from sidecar if available
    if metadata and "SamplingFrequency" in metadata:
        snirf = snirf._replace(
            sampling_frequency=float(metadata["SamplingFrequency"])
        )

    # Read events
    events_path = nirs_dir / f"{stem}_events.tsv"
    events = _read_events_tsv(events_path)

    # Read optodes
    optodes_path = nirs_dir / f"{stem}_optodes.tsv"
    optodes = _read_tsv(optodes_path)

    # Read channels
    channels_path = nirs_dir / f"{stem}_channels.tsv"
    channels = _read_tsv(channels_path)

    # Read coordsystem
    coord_path = nirs_dir / f"{stem}_coordsystem.json"
    coordsystem = _read_json(coord_path)

    task_name = metadata.get("TaskName") if metadata else None

    return BidsNirsRun(
        snirf=snirf,
        task_name=task_name,
        events=events,
        optodes_tsv=optodes,
        channels_tsv=channels,
        coordsystem=coordsystem,
        metadata=metadata,
    )


# =============================================================================
# Conversion to dot-jax arrays
# =============================================================================


def snirf_to_dot_jax(snirf_data, wavelength=None):
    """Extract JAX arrays from SnirfData for dot-jax forward/inverse.

    Parameters
    ----------
    snirf_data : SnirfData
        Output of read_snirf or load_bids_nirs.
    wavelength : float, optional
        Select channels for a specific wavelength [nm].
        If None, returns all channels.

    Returns
    -------
    dict with keys:
        srcpos : jnp.ndarray (n_src, 3) — source positions [mm]
        detpos : jnp.ndarray (n_det, 3) — detector positions [mm]
        wavelengths : jnp.ndarray (n_wl,) — wavelength list [nm]
        data : jnp.ndarray (n_time, n_chan) — measurement data
        channel_src : jnp.ndarray (n_chan,) — 0-based source index per channel
        channel_det : jnp.ndarray (n_chan,) — 0-based detector index per channel
        channel_wl : jnp.ndarray (n_chan,) — wavelength [nm] per channel
        fs : float — sampling frequency [Hz]
    """
    ml = snirf_data.measurement_list
    wl_array = snirf_data.wavelengths

    # Map 1-based SNIRF indices to 0-based
    src_idx = ml.source_index - 1
    det_idx = ml.detector_index - 1
    wl_per_chan = wl_array[ml.wavelength_index - 1]

    # Filter by wavelength if requested
    if wavelength is not None:
        mask = np.isclose(wl_per_chan, wavelength, atol=5.0)
        if not np.any(mask):
            available = np.unique(wl_per_chan)
            raise ValueError(
                f"No channels at {wavelength}nm. Available: {available}"
            )
        data = snirf_data.data[:, mask]
        src_idx = src_idx[mask]
        det_idx = det_idx[mask]
        wl_per_chan = wl_per_chan[mask]
    else:
        data = snirf_data.data

    return {
        "srcpos": jnp.array(snirf_data.source_pos, dtype=jnp.float64),
        "detpos": jnp.array(snirf_data.detector_pos, dtype=jnp.float64),
        "wavelengths": jnp.array(wl_array, dtype=jnp.float64),
        "data": jnp.array(data, dtype=jnp.float64),
        "channel_src": jnp.array(src_idx, dtype=jnp.int32),
        "channel_det": jnp.array(det_idx, dtype=jnp.int32),
        "channel_wl": jnp.array(wl_per_chan, dtype=jnp.float64),
        "fs": snirf_data.sampling_frequency,
    }


# =============================================================================
# TSV / JSON helpers
# =============================================================================


def _read_json(path):
    """Read JSON file, return None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _read_tsv(path):
    """Read TSV into a dict of lists, return None if missing."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return None
    return {k: [r[k] for r in rows] for k in rows[0]}


def _read_events_tsv(path):
    """Read BIDS events.tsv into (n_events, 3) array [onset, duration, trial_idx]."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return None

    trial_types = sorted(set(r.get("trial_type", "unknown") for r in rows))
    type_map = {t: i for i, t in enumerate(trial_types)}

    events = np.zeros((len(rows), 3))
    for i, r in enumerate(rows):
        events[i, 0] = float(r.get("onset", 0))
        events[i, 1] = float(r.get("duration", 0))
        events[i, 2] = type_map.get(r.get("trial_type", "unknown"), 0)

    return events
