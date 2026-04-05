"""SNIRF, BIDS-fNIRS, and JMesh/NeuroJSON I/O for dot-jax.

Reads SNIRF files (HDF5), BIDS-fNIRS directory layouts, and
JMesh (NeuroJSON) mesh files into JAX-compatible arrays. Uses h5py
directly — no external SNIRF library required.

SNIRF spec: https://github.com/fNIRS/snirf
BIDS-fNIRS: https://bids-specification.readthedocs.io/en/stable/
             modality-specific-files/near-infrared-spectroscopy.html
JMesh spec: https://github.com/NeuroJSON/jmesh
NeuroJSON:  https://neurojson.io

Functions:
    read_snirf: Read a .snirf file into a SnirfData container
    load_bids_nirs: Load a BIDS-fNIRS subject/session directory
    snirf_to_dot_jax: Extract arrays needed for forward_cw / reconstruct_mua
    read_jmesh: Read a JMesh file (local or URL) into node/elem arrays
    fetch_neurojson: Fetch a document from the NeuroJSON CouchDB API
    get_mcx_optical_properties: Get tissue optical properties from MCX benchmarks
    load_brain_mesh: Load a pre-built brain mesh from BrainMeshLibrary
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


# =============================================================================
# JMesh / NeuroJSON I/O
# =============================================================================

_JMESH_DTYPE_MAP = {
    "single": np.float32,
    "double": np.float64,
    "int32": np.int32,
    "uint8": np.uint8,
    "uint32": np.uint32,
    "int64": np.int64,
}


def _resolve_jmesh_data(data_field):
    """Resolve a JMesh data field: inline string, _DataLink_ URL, or raw bytes."""
    import subprocess

    if isinstance(data_field, str):
        return data_field  # inline base64 string
    elif isinstance(data_field, dict):
        if "_DataLink_" in data_field:
            url = data_field["_DataLink_"]
            result = subprocess.run(
                ["curl", "-skL", url],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to fetch {url}")
            return result.stdout  # raw bytes
        raise ValueError(f"Unknown JMesh data dict: {list(data_field.keys())}")
    elif isinstance(data_field, bytes):
        return data_field
    raise TypeError(f"Cannot resolve JMesh data of type {type(data_field)}")


def _decode_jmesh_array(data_field, info):
    """Decode a JMesh compressed+encoded array.

    JMesh arrays are base64-encoded, optionally zlib-compressed, with
    metadata in a companion dict. Data may also be a _DataLink_ URL
    to an external binary resource.
    """
    import base64
    import zlib

    resolved = _resolve_jmesh_data(data_field)

    # If we got raw bytes from a _DataLink_, it's already binary
    if isinstance(resolved, bytes):
        raw = resolved
    else:
        raw = base64.b64decode(resolved)

    zip_type = info.get("_ArrayZipType_", info.get("ArrayZipType", "")).lower()
    if zip_type == "zlib":
        raw = zlib.decompress(raw)

    type_key = info.get("_ArrayType_", info.get("ArrayType", "single"))
    dtype = _JMESH_DTYPE_MAP.get(type_key, np.float32)

    shape = info.get("_ArraySize_", info.get("ArraySize", []))
    if isinstance(shape, (int, float)):
        shape = [int(shape)]
    else:
        shape = [int(s) for s in shape]

    arr = np.frombuffer(raw, dtype=dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def read_jmesh(source):
    """Read a JMesh mesh file (local path or parsed dict).

    Decodes MeshNode and MeshElem arrays from JMesh format.
    JMesh stores arrays as base64+zlib compressed data with _DataInfo_
    metadata describing dtype and shape.

    Parameters
    ----------
    source : str, Path, or dict
        Path to a .jmesh/.json file, or an already-parsed dict.

    Returns
    -------
    result : dict
        Keys: 'node' (N, 3) float64 — node coordinates,
              'elem' (M, 4+) int32 — element connectivity (0-based),
              'elem_labels' (M,) int32 — tissue labels (if column 5 exists).
    """
    if isinstance(source, dict):
        data = source
    else:
        path = Path(source)
        with open(path) as f:
            data = json.load(f)

    result = {}

    # Decode MeshNode
    if "MeshNode" in data:
        nd = data["MeshNode"]
        if isinstance(nd, dict) and "_ArrayZipData_" in nd:
            node = _decode_jmesh_array(nd["_ArrayZipData_"], nd)
        elif isinstance(nd, str):
            info = data.get("MeshNode_DataInfo_", {})
            node = _decode_jmesh_array(nd, info)
        else:
            node = np.array(nd, dtype=np.float64)
        result["node"] = node.astype(np.float64)

    # Decode MeshElem
    if "MeshElem" in data:
        ed = data["MeshElem"]
        if isinstance(ed, dict) and "_ArrayZipData_" in ed:
            elem = _decode_jmesh_array(ed["_ArrayZipData_"], ed)
        elif isinstance(ed, str):
            info = data.get("MeshElem_DataInfo_", {})
            elem = _decode_jmesh_array(ed, info)
        else:
            elem = np.array(ed, dtype=np.int32)
        elem = elem.astype(np.int32)

        # JMesh uses 1-based indexing; convert to 0-based
        # Column 5 (if present) is the tissue label, not a node index
        if elem.shape[1] >= 5:
            result["elem_labels"] = elem[:, 4].copy()
            result["elem"] = elem[:, :4] - 1  # 0-based
        else:
            result["elem"] = elem - 1  # 0-based

    return result


def fetch_neurojson(database, document, cache=True):
    """Fetch a document from the NeuroJSON CouchDB API.

    Parameters
    ----------
    database : str — database name (e.g. 'brainmeshlibrary', 'mcx').
    document : str — document ID (e.g. 'colin27', 'BrainWeb--Subject04--head--tet').
    cache : bool — cache to ~/.cache/dot-jax/neurojson/.

    Returns
    -------
    data : dict — parsed JSON document.
    """
    cache_dir = Path.home() / ".cache/dot-jax/neurojson"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{database}_{document}.json"

    if cache and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    import subprocess
    url = f"https://neurojson.io:7777/{database}/{document}"
    result = subprocess.run(
        ["curl", "-sk", url],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {url}: {result.stderr}")

    data = json.loads(result.stdout)

    if cache:
        with open(cache_file, "w") as f:
            json.dump(data, f)

    return data


def get_mcx_optical_properties(benchmark="colin27"):
    """Get tissue optical properties from an MCX benchmark.

    Parameters
    ----------
    benchmark : str — MCX benchmark name (e.g. 'colin27', '4layer_head').

    Returns
    -------
    properties : dict
        Maps tissue label (int) to dict with keys:
        'mua', 'mus', 'g', 'n', 'musp', 'name'.
    """
    data = fetch_neurojson("mcx", benchmark)
    media = data["Domain"]["Media"]

    # Standard MCX tissue names per benchmark
    _names = {
        "colin27": {0: "background", 1: "scalp", 2: "skull", 3: "CSF",
                    4: "gray_matter", 5: "white_matter", 6: "air"},
        "4layer_head": {0: "background", 1: "scalp", 2: "CSF",
                        3: "gray_matter", 4: "white_matter"},
    }
    names = _names.get(benchmark, {})

    props = {}
    for i, m in enumerate(media):
        musp = m["mus"] * (1 - m["g"]) if m["mus"] > 0 else 0.0
        props[i] = {
            "mua": m["mua"],
            "mus": m["mus"],
            "g": m["g"],
            "n": m["n"],
            "musp": musp,
            "name": names.get(i, f"tissue_{i}"),
        }
    return props


def load_brain_mesh(atlas="BrainWeb", subject="Subject04"):
    """Load a pre-built brain mesh from the NeuroJSON BrainMeshLibrary.

    Parameters
    ----------
    atlas : str — 'BrainWeb' or 'NDMRI'.
    subject : str — subject/age identifier (e.g. 'Subject04', '20-24Years3T').

    Returns
    -------
    mesh : FEMMesh — tetrahedral mesh ready for forward modelling.
    tissue_labels : (ne,) int — tissue label per element
        (1=scalp, 2=skull, 3=CSF, 4=GM, 5=WM).
    """
    from .mesh import FEMMesh

    doc_id = f"{atlas}--{subject}--head--tet"
    data = fetch_neurojson("brainmeshlibrary", doc_id)
    parsed = read_jmesh(data)

    node = parsed["node"]
    elem = parsed["elem"]
    labels = parsed.get("elem_labels", np.ones(len(elem), dtype=np.int32))

    mesh = FEMMesh.create(
        jnp.array(node, dtype=jnp.float64),
        jnp.array(elem, dtype=jnp.int32),
    )

    return mesh, labels
