"""Real-time fNIRS reconstruction pipeline.

Pre-computes the Jacobian pseudoinverse at startup, then processes
each incoming frame as a single matrix-vector multiply. Designed for
real-time streaming from Kernel Flow or other fNIRS devices.

Target: <5 ms per frame on GPU (4.75 Hz Kernel Flow → 210 ms budget).

Classes:
    RealtimePipeline: Pre-computed Jinv reconstruction engine
    EpochAccumulator: Event-locked HbO averaging for cognitive experiments
"""

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from .assembly import assemble_system_cw
from .forward import assemble_rhs, get_detector_values
from .hemodynamics import spectral_unmix
from .property import extinction
from .recon import solve_dual


class RealtimePipeline:
    """Pre-computed fast reconstruction for real-time fNIRS.

    At init: assembles the system matrix, solves Green's functions,
    builds Rytov-normalised Jacobians, and pre-computes their
    regularised pseudoinverses. Each subsequent frame is a single
    matrix-vector multiply per wavelength + spectral unmixing.

    Parameters
    ----------
    mesh : FEMMesh
    srcpos : (n_src, 3) — source positions.
    detpos : (n_det, 3) — detector positions.
    mua : float — baseline absorption coefficient.
    musp : float — baseline reduced scattering coefficient.
    wavelengths : list of float — wavelengths in nm.
    n_in : float — tissue refractive index.
    n_out : float — external refractive index.
    sd_range : (float, float) — source-detector distance range to keep (mm).
    reg : float — regularisation parameter for dual solver.
    """

    def __init__(self, mesh, srcpos, detpos, mua=0.01, musp=1.0,
                 wavelengths=(690.0, 830.0), n_in=1.37, n_out=1.0,
                 sd_range=(8.0, 55.0), reg=0.01):
        import lineax as lx

        self.mesh = mesh
        self.nn = mesh.nn
        self.n_src = srcpos.shape[0]
        self.n_det = detpos.shape[0]
        self.wavelengths = list(wavelengths)
        self._frame_count = 0
        self._prev_frame = None

        # Assemble system matrix
        A = assemble_system_cw(mesh, mua, musp, n_in, n_out)
        operator = lx.MatrixLinearOperator(A)

        def solve_col(b):
            return lx.linear_solve(operator, b, solver=lx.LU()).value

        # Source/detector RHS and Green's functions
        rhs_src = assemble_rhs(mesh, srcpos)
        rhs_det = assemble_rhs(mesh, detpos)
        phi_src = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_src)
        phi_det = jax.vmap(solve_col, in_axes=1, out_axes=1)(rhs_det)
        nvol = mesh.nvol

        # SD distances for all possible (src, det) pairs
        srcpos_np = np.array(srcpos)
        detpos_np = np.array(detpos)

        # Build Jacobian and Jinv per wavelength
        # For real-time, we build a generic Jacobian over all src-det pairs
        # within the SD range. The channel mapping from the device tells us
        # which rows to use for each frame.
        self.Jinv = {}
        self.n_channels = {}
        self._channel_maps = {}

        for w_idx in range(len(wavelengths)):
            # All possible src-det pairs
            rows = []
            ch_map = []
            for s in range(self.n_src):
                for d in range(self.n_det):
                    dist = np.sqrt(np.sum((srcpos_np[s] - detpos_np[d]) ** 2))
                    if sd_range[0] <= dist <= sd_range[1]:
                        raw = -phi_src[:, s] * phi_det[:, d] * nvol
                        pred = rhs_det[:, d].T @ phi_src[:, s]
                        rows.append(raw / jnp.maximum(jnp.abs(pred), 1e-30))
                        ch_map.append((s, d))

            if rows:
                J = jnp.stack(rows, axis=0)
                # Pre-compute pseudoinverse via dual formulation
                JJt = J @ J.T
                norm_scale = jnp.sqrt(jnp.linalg.norm(JJt))
                H_reg = JJt + reg * norm_scale * jnp.eye(J.shape[0])
                H_reg = 0.5 * (H_reg + H_reg.T)
                # Jinv = J^T (JJ^T + reg*I)^{-1}
                H_inv = jnp.linalg.inv(H_reg)
                self.Jinv[w_idx] = J.T @ H_inv  # (nn, n_ch)
            else:
                self.Jinv[w_idx] = jnp.zeros((self.nn, 0))

            self.n_channels[w_idx] = len(rows)
            self._channel_maps[w_idx] = ch_map

        # Extinction matrix for spectral unmixing
        wv_int = [int(w) for w in wavelengths]
        self._E_inv = jnp.linalg.pinv(extinction(wv_int, ["hbo", "hbr"]))

    def process_frame(self, od_frame):
        """Process a single frame of OD data.

        Parameters
        ----------
        od_frame : dict mapping wavelength_index → (n_channels,) OD array.

        Returns
        -------
        hbo : (nn,) — oxyhemoglobin change at this frame.
        hbr : (nn,) — deoxyhemoglobin change at this frame.
        """
        self._frame_count += 1

        # Reconstruct delta_mua per wavelength
        delta_mua = []
        for w_idx in range(len(self.wavelengths)):
            od = od_frame[w_idx]
            dmua = self.Jinv[w_idx] @ od  # single matrix-vector multiply
            delta_mua.append(dmua)

        # Spectral unmixing: E_inv @ [dmua_wv1, dmua_wv2]
        mua_stack = jnp.stack(delta_mua, axis=0)  # (n_wv, nn)
        hb = self._E_inv @ mua_stack  # (2, nn)

        return hb[0], hb[1]  # hbo, hbr

    def get_quality_metrics(self):
        """Return current quality metrics."""
        return {
            "frame_count": self._frame_count,
            "n_channels": {w: self.n_channels[w] for w in range(len(self.wavelengths))},
            "nn": self.nn,
        }


class EpochAccumulator:
    """Event-locked averaging for cognitive neuroscience experiments.

    Accumulates HbO frames into epochs triggered by experiment events,
    computes running average and standard deviation.

    Parameters
    ----------
    n_nodes : int — number of mesh nodes.
    epoch_samples : int — number of frames per epoch after event onset.
    fs : float — sampling frequency (Hz).
    """

    def __init__(self, n_nodes, epoch_samples=30, fs=1.0):
        self.n_nodes = n_nodes
        self.epoch_samples = epoch_samples
        self.fs = fs
        self._buffer = deque(maxlen=epoch_samples * 10)
        self._epochs = []
        self._current_epoch = None
        self._epoch_counter = 0

    @property
    def n_epochs(self):
        return len(self._epochs)

    def add_frame(self, hbo, event=False):
        """Add a frame to the buffer, optionally starting a new epoch.

        Parameters
        ----------
        hbo : (n_nodes,) — HbO values at this frame.
        event : bool — True if an experiment event occurred at this frame.
        """
        hbo_np = np.array(hbo)
        self._buffer.append(hbo_np)

        # If there's an active epoch being collected, add to it
        if self._current_epoch is not None:
            self._current_epoch.append(hbo_np)
            if len(self._current_epoch) >= self.epoch_samples:
                self._epochs.append(np.stack(self._current_epoch, axis=0))
                self._current_epoch = None

        # Start a new epoch on event
        if event:
            self._current_epoch = [hbo_np]
            self._epoch_counter += 1

    def get_average(self):
        """Get the running epoch average and standard deviation.

        Returns
        -------
        (avg, std) : tuple of (epoch_samples, n_nodes) arrays, or None if no epochs.
        """
        if not self._epochs:
            return None

        all_epochs = np.stack(self._epochs, axis=0)  # (n_epochs, epoch_samples, n_nodes)
        avg = jnp.array(np.mean(all_epochs, axis=0))
        std = jnp.array(np.std(all_epochs, axis=0))
        return avg, std
