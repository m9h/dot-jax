"""Physiologically-constrained DOT inference — RED-GREEN TDD.

Tests that coupling vpjax's Balloon-Windkessel model with dot-jax's
forward model improves reconstruction quality over frame-independent
inversion. This is the Diamond et al. (2006) approach: a state-space
model where vascular physiology provides the temporal dynamics and
the DOT Jacobian provides the observation model.

Pipeline:
    neural_activity(t) → Balloon ODE → [HbO(t), HbR(t)] per node
    [HbO, HbR] → extinction coefficients → delta_mua(t)
    delta_mua(t) → J @ delta_mua → predicted OD(t)

The physiological filter takes noisy frame-by-frame reconstructions
and fits Balloon dynamics per node, producing:
    - Smoothed, physiologically plausible HbO/HbR time courses
    - Estimated neural activity time course per node
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import Delaunay

jax.config.update("jax_enable_x64", True)

from vpjax.hemodynamics.balloon import solve_balloon
from vpjax._types import BalloonParams, BalloonState
from vpjax.hemodynamics.optics import to_optical_properties

from dot_jax.mesh import FEMMesh
from dot_jax.spectral import compute_jacobian_mua
from dot_jax.recon import reconstruct_image
from dot_jax.property import extinction
from dot_jax.hemodynamics import spectral_unmix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WAVELENGTHS = jnp.array([760.0, 850.0])
DT = 0.1  # 10 Hz, typical fNIRS
HbO_0 = 0.060  # mM baseline oxy-hemoglobin
HbR_0 = 0.040  # mM baseline deoxy-hemoglobin
HbT_0 = HbO_0 + HbR_0


@pytest.fixture(scope="module")
def slab():
    """Slab phantom."""
    nx, ny, nz, dx = 8, 8, 5, 5.0
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dx, ny)
    z = np.linspace(0, (nz - 1) * dx, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    node = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    tri = Delaunay(node)
    return FEMMesh.create(node, tri.simplices.astype(np.int32))


@pytest.fixture(scope="module")
def optodes(slab):
    z_max = float(np.asarray(slab.node)[:, 2].max())
    src_x = np.linspace(8.75, 26.25, 3)
    src_y = np.linspace(8.75, 26.25, 3)
    sxx, syy = np.meshgrid(src_x, src_y)
    srcpos = np.column_stack([sxx.ravel(), syy.ravel(), np.full(9, z_max)])
    det_x = np.linspace(4.375, 30.625, 4)
    det_y = np.linspace(4.375, 30.625, 4)
    dxx, dyy = np.meshgrid(det_x, det_y)
    detpos = np.column_stack([dxx.ravel(), dyy.ravel(), np.full(16, z_max)])
    return jnp.array(srcpos), jnp.array(detpos)


@pytest.fixture(scope="module")
def jacobians(slab, optodes):
    """Jacobians at 760nm and 850nm."""
    srcpos, detpos = optodes
    ext = extinction(WAVELENGTHS, ["hbo", "hbr"])
    # Use baseline mua at each wavelength
    c_bg = jnp.array([HbO_0 * 1000, HbR_0 * 1000])  # uM
    mua_bg = c_bg @ ext.T  # (2,) in 1/mm
    jacs = []
    for w in range(2):
        J = compute_jacobian_mua(slab, float(mua_bg[w]), 1.0, srcpos, detpos)
        jacs.append(J)
    return jacs


@pytest.fixture(scope="module")
def balloon_params():
    return BalloonParams()


@pytest.fixture(scope="module")
def synthetic_experiment(slab, jacobians, balloon_params):
    """Generate synthetic block-design DOT data driven by Balloon physiology.

    Block design: 10s ON / 20s OFF × 2 blocks = 60s total at 10 Hz.
    Neural activity at a focal region drives the Balloon model.
    The resulting HbO/HbR are converted to delta_mua and projected
    through the Jacobian to produce synthetic OD measurements.

    Returns dict with ground truth and noisy measurements.
    """
    node = np.asarray(slab.node)
    nn = slab.nn
    z_max = node[:, 2].max()
    center = np.array([17.5, 17.5, z_max - 5.0])
    dist = np.linalg.norm(node - center, axis=1)
    spatial_mask = np.exp(-dist ** 2 / (2 * 5.0 ** 2))  # Gaussian blob

    n_time = int(60.0 / DT)  # 600 frames
    ext = np.asarray(extinction(WAVELENGTHS, ["hbo", "hbr"]))

    # Block-design stimulus
    stimulus = np.zeros(n_time)
    stimulus[int(5.0 / DT):int(15.0 / DT)] = 0.3  # block 1 (amplitude 0.3)
    stimulus[int(35.0 / DT):int(45.0 / DT)] = 0.3  # block 2

    # Run Balloon model at the activation center
    ts, traj = solve_balloon(balloon_params, jnp.array(stimulus), dt=DT)

    # Convert Balloon state → HbO/HbR (absolute, mM)
    hbo_center = HbT_0 * np.asarray(traj.v) - HbR_0 * np.asarray(traj.q)
    hbr_center = HbR_0 * np.asarray(traj.q)

    # Delta from baseline
    delta_hbo_center = hbo_center - HbO_0  # (n_time,)
    delta_hbr_center = hbr_center - HbR_0

    # Spatial distribution: delta per node = center_timecourse × spatial_mask
    delta_hbo = delta_hbo_center[:, None] * spatial_mask[None, :]  # (n_time, nn)
    delta_hbr = delta_hbr_center[:, None] * spatial_mask[None, :]

    # Convert to delta_mua at each wavelength (mM → 1/mm via extinction)
    # delta_mua[w] = ext[w,0]*delta_hbo*1000 + ext[w,1]*delta_hbr*1000
    # (extinction is in 1/(mm*uM), concentrations in mM = 1000 uM)
    data_per_wl = []
    for w in range(2):
        delta_mua_w = (ext[w, 0] * delta_hbo * 1000
                       + ext[w, 1] * delta_hbr * 1000)  # (n_time, nn)
        # Project through Jacobian: OD = J @ delta_mua per frame
        J = np.asarray(jacobians[w])
        data_w = np.array([J @ delta_mua_w[t] for t in range(n_time)])
        data_per_wl.append(data_w)

    # Add noise
    rng = np.random.default_rng(42)
    snr_db = 20.0
    noisy_per_wl = []
    for w in range(2):
        signal_power = np.mean(data_per_wl[w] ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), data_per_wl[w].shape)
        noisy_per_wl.append(data_per_wl[w] + noise)

    return {
        "stimulus": stimulus,
        "delta_hbo_true": delta_hbo,
        "delta_hbr_true": delta_hbr,
        "data_clean": data_per_wl,
        "data_noisy": noisy_per_wl,
        "spatial_mask": spatial_mask,
        "n_time": n_time,
    }


# ---------------------------------------------------------------------------
# Helper: frame-by-frame reconstruction (baseline to beat)
# ---------------------------------------------------------------------------


def _frame_by_frame_recon(jacobians, data_per_wl, n_time, ext_matrix):
    """Reconstruct HbO/HbR per frame using pre-computed Jinv (fast).

    Uses solve_dual to pre-compute the regularized pseudoinverse,
    then each frame is a single matrix-vector multiply.
    """
    from dot_jax.recon import solve_dual

    # Pre-compute Jinv per wavelength (once)
    jinv = []
    for w in range(2):
        J = np.asarray(jacobians[w])
        # Build Jinv: nn × n_meas
        JJt = J @ J.T
        norm_scale = np.sqrt(np.linalg.norm(JJt))
        H_reg = JJt + 0.01 * norm_scale * np.eye(J.shape[0])
        H_reg = 0.5 * (H_reg + H_reg.T)
        H_inv = np.linalg.inv(H_reg)
        jinv.append(J.T @ H_inv)

    # Spectral unmixing matrix
    E_inv = np.asarray(jnp.linalg.pinv(ext_matrix))

    recon_hbo = np.zeros((n_time, jacobians[0].shape[1]))
    recon_hbr = np.zeros((n_time, jacobians[0].shape[1]))

    for t in range(n_time):
        dmua = []
        for w in range(2):
            dmua.append(jinv[w] @ data_per_wl[w][t])
        # Unmix: E_inv @ [dmua_750, dmua_850] → [HbO, HbR]
        mua_stack = np.stack(dmua, axis=0)  # (2, nn)
        hb = E_inv @ mua_stack  # (2, nn)
        recon_hbo[t] = hb[0]
        recon_hbr[t] = hb[1]

    return recon_hbo, recon_hbr


# ---------------------------------------------------------------------------
# Tests: synthetic data generation
# ---------------------------------------------------------------------------


class TestSyntheticGeneration:
    def test_balloon_response_shape(self, synthetic_experiment):
        """Synthetic data should have expected dimensions."""
        exp = synthetic_experiment
        assert exp["delta_hbo_true"].shape == (exp["n_time"], 320)
        assert len(exp["data_clean"]) == 2

    def test_activation_present(self, synthetic_experiment):
        """HbO should increase and HbR decrease during stimulation."""
        exp = synthetic_experiment
        # During first block (5-15s), HbO should peak
        on_slice = slice(int(8.0 / DT), int(14.0 / DT))
        off_slice = slice(int(25.0 / DT), int(32.0 / DT))

        hbo_on = exp["delta_hbo_true"][on_slice].max()
        hbo_off = np.abs(exp["delta_hbo_true"][off_slice]).max()
        assert hbo_on > hbo_off, "HbO should be larger during ON block"

    def test_hbo_hbr_opposite_signs(self, synthetic_experiment):
        """During activation, HbO up and HbR down (anti-correlated)."""
        exp = synthetic_experiment
        peak_frame = int(12.0 / DT)
        center_node = np.argmax(exp["spatial_mask"])

        assert exp["delta_hbo_true"][peak_frame, center_node] > 0
        assert exp["delta_hbr_true"][peak_frame, center_node] < 0


# ---------------------------------------------------------------------------
# Tests: physiological filter (RED — function does not exist yet)
# ---------------------------------------------------------------------------


class TestPhysiologicalFilter:
    """Tests for dot_jax.inference.physiological_filter.

    This function takes noisy frame-by-frame HbO/HbR reconstructions
    and smooths them using Balloon-Windkessel dynamics per node.
    """

    def test_import(self):
        """physiological_filter should be importable."""
        from dot_jax.inference import physiological_filter

    def test_returns_correct_shapes(self, slab, jacobians, synthetic_experiment,
                                    balloon_params):
        """Output should have (n_time, nn) for HbO, HbR, and neural."""
        from dot_jax.inference import physiological_filter

        exp = synthetic_experiment
        ext = extinction(WAVELENGTHS, ["hbo", "hbr"])

        # Get noisy frame-by-frame recon
        hbo_noisy, hbr_noisy = _frame_by_frame_recon(
            jacobians, exp["data_noisy"], exp["n_time"], ext,
        )

        result = physiological_filter(
            hbo_noisy, hbr_noisy, DT, balloon_params,
        )

        assert result.hbo.shape == (exp["n_time"], slab.nn)
        assert result.hbr.shape == (exp["n_time"], slab.nn)
        assert result.neural.shape == (exp["n_time"], slab.nn)

    def test_smoother_than_frame_by_frame(self, slab, jacobians,
                                          synthetic_experiment, balloon_params):
        """Physiological filter should produce smoother time courses.

        The temporal derivative variance of the filtered HbO should be
        smaller than the frame-by-frame reconstruction, because the
        Balloon dynamics enforce physiological smoothness.
        """
        from dot_jax.inference import physiological_filter

        exp = synthetic_experiment
        ext = extinction(WAVELENGTHS, ["hbo", "hbr"])

        hbo_noisy, hbr_noisy = _frame_by_frame_recon(
            jacobians, exp["data_noisy"], exp["n_time"], ext,
        )

        result = physiological_filter(
            hbo_noisy, hbr_noisy, DT, balloon_params,
        )

        # Temporal derivative variance at the activation center
        center = np.argmax(exp["spatial_mask"])
        deriv_noisy = np.diff(hbo_noisy[:, center])
        deriv_filtered = np.diff(np.asarray(result.hbo[:, center]))

        assert deriv_filtered.var() < deriv_noisy.var(), (
            f"Filtered deriv var ({deriv_filtered.var():.2e}) should be less "
            f"than noisy ({deriv_noisy.var():.2e})"
        )

    def test_closer_to_ground_truth_low_snr(self, slab, jacobians,
                                           synthetic_experiment, balloon_params):
        """At low SNR, physiological filter should beat frame-by-frame.

        At high SNR the linear inverse is already near-perfect, so the
        EKF's value emerges under challenging conditions: low SNR where
        the temporal prior helps denoise.
        """
        from dot_jax.inference import physiological_filter

        exp = synthetic_experiment
        ext = extinction(WAVELENGTHS, ["hbo", "hbr"])

        # Add heavy noise (0 dB SNR) to the clean data
        rng = np.random.default_rng(123)
        noisy_heavy = []
        for w in range(2):
            signal_power = np.mean(exp["data_clean"][w] ** 2)
            noise = rng.normal(0, np.sqrt(signal_power), exp["data_clean"][w].shape)
            noisy_heavy.append(exp["data_clean"][w] + noise)

        hbo_noisy, hbr_noisy = _frame_by_frame_recon(
            jacobians, noisy_heavy, exp["n_time"], ext,
        )

        result = physiological_filter(
            hbo_noisy, hbr_noisy, DT, balloon_params,
        )

        true_hbo = exp["delta_hbo_true"]
        center = np.argmax(exp["spatial_mask"])

        r_noisy = float(np.corrcoef(hbo_noisy[:, center],
                                     true_hbo[:, center])[0, 1])
        r_filtered = float(np.corrcoef(np.asarray(result.hbo[:, center]),
                                        true_hbo[:, center])[0, 1])

        assert r_filtered > r_noisy, (
            f"Filtered correlation ({r_filtered:.3f}) should exceed "
            f"noisy ({r_noisy:.3f}) at 0 dB SNR"
        )

    def test_recovers_neural_timing(self, slab, jacobians,
                                    synthetic_experiment, balloon_params):
        """Estimated neural activity should be larger during ON blocks."""
        from dot_jax.inference import physiological_filter

        exp = synthetic_experiment
        ext = extinction(WAVELENGTHS, ["hbo", "hbr"])

        hbo_noisy, hbr_noisy = _frame_by_frame_recon(
            jacobians, exp["data_noisy"], exp["n_time"], ext,
        )

        result = physiological_filter(
            hbo_noisy, hbr_noisy, DT, balloon_params,
        )

        center = np.argmax(exp["spatial_mask"])
        neural = np.asarray(result.neural[:, center])

        # ON blocks: 5-15s and 35-45s
        on1 = slice(int(8.0 / DT), int(14.0 / DT))
        off = slice(int(22.0 / DT), int(32.0 / DT))

        mean_on = np.abs(neural[on1]).mean()
        mean_off = np.abs(neural[off]).mean()

        assert mean_on > mean_off, (
            f"Neural ON ({mean_on:.4f}) should exceed OFF ({mean_off:.4f})"
        )

    def test_hbo_hbr_anti_correlated(self, slab, jacobians,
                                     synthetic_experiment, balloon_params):
        """Filtered HbO and HbR should be anti-correlated at active nodes.

        The Balloon model enforces this: when CBF increases, HbO goes up
        and HbR goes down. Frame-by-frame reconstruction may not
        preserve this constraint.
        """
        from dot_jax.inference import physiological_filter

        exp = synthetic_experiment
        ext = extinction(WAVELENGTHS, ["hbo", "hbr"])

        hbo_noisy, hbr_noisy = _frame_by_frame_recon(
            jacobians, exp["data_noisy"], exp["n_time"], ext,
        )

        result = physiological_filter(
            hbo_noisy, hbr_noisy, DT, balloon_params,
        )

        center = np.argmax(exp["spatial_mask"])
        r = float(np.corrcoef(np.asarray(result.hbo[:, center]),
                               np.asarray(result.hbr[:, center]))[0, 1])

        assert r < 0, (
            f"HbO-HbR correlation should be negative (got r={r:.3f})"
        )
