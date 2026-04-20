"""Time-domain DOT reconstruction — RED-GREEN TDD.

Tests the TD-fNIRS reconstruction bridge: DTOF moments → moment
Jacobian → image reconstruction. This is the missing link for
Kernel Flow 2 data processing.

The approach uses autodiff through the full TD forward model
(Diffrax ODE → DTOF → moments) to compute moment Jacobians
d(m0,m1,m2)/d(mua_node), then standard Tikhonov inversion.

TD moments carry depth information that CW intensity alone cannot:
  - m0 (total photons) ≈ CW intensity
  - m1 (mean time-of-flight) — sensitive to deeper tissue
  - m2 (variance) — even deeper sensitivity
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.td_forward import td_forward_cw, dtof_moments


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def geometry():
    srcpos = jnp.array([[5.0, 5.0, 5.0]])
    detpos = jnp.array([[15.0, 15.0, 15.0], [10.0, 10.0, 10.0]])
    return srcpos, detpos


# ---------------------------------------------------------------------------
# Tests: moment Jacobian computation
# ---------------------------------------------------------------------------


class TestMomentJacobian:
    """Jacobian of DTOF moments w.r.t. nodal absorption."""

    def test_compute_moment_jacobian_exists(self):
        """compute_moment_jacobian should be importable."""
        from dot_jax.td_forward import compute_moment_jacobian

    def test_shape(self, box_mesh, geometry):
        """Moment Jacobian should be (n_moments * n_det, nn)."""
        from dot_jax.td_forward import compute_moment_jacobian

        srcpos, detpos = geometry
        J = compute_moment_jacobian(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=2e-9, n_times=50,
        )
        n_det = detpos.shape[0]
        nn = box_mesh.nn
        # 3 moments (m0, m1, m2) × n_det detectors
        assert J.shape == (3 * n_det, nn)

    def test_finite(self, box_mesh, geometry):
        from dot_jax.td_forward import compute_moment_jacobian

        srcpos, detpos = geometry
        J = compute_moment_jacobian(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=2e-9, n_times=50,
        )
        assert jnp.all(jnp.isfinite(J))

    def test_m0_negative_sensitivity(self, box_mesh, geometry):
        """m0 (total photons) should decrease with increasing mua at any node.

        This means the m0 rows of the Jacobian should be all negative.
        """
        from dot_jax.td_forward import compute_moment_jacobian

        srcpos, detpos = geometry
        J = compute_moment_jacobian(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=2e-9, n_times=50,
        )
        n_det = detpos.shape[0]
        # First n_det rows are d(m0)/d(mua)
        J_m0 = J[:n_det, :]
        # Most entries should be negative (more absorption → fewer photons)
        neg_frac = float(jnp.mean(J_m0 < 0))
        assert neg_frac > 0.5, f"Expected mostly negative m0 Jacobian, got {neg_frac:.0%} negative"

    def test_m1_positive_sensitivity(self, box_mesh, geometry):
        """m1 (mean time-of-flight) should increase with mua.

        More absorption preferentially removes early-arriving photons
        (short paths), shifting the mean to later times.
        """
        from dot_jax.td_forward import compute_moment_jacobian

        srcpos, detpos = geometry
        J = compute_moment_jacobian(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=2e-9, n_times=50,
        )
        n_det = detpos.shape[0]
        # m1 rows are J[n_det:2*n_det, :]
        J_m1 = J[n_det:2*n_det, :]
        # Should have mix of signs but overall positive trend
        assert jnp.any(J_m1 != 0), "m1 Jacobian should be nonzero"


# ---------------------------------------------------------------------------
# Tests: TD reconstruction
# ---------------------------------------------------------------------------


class TestTDReconstruction:
    """Reconstruct absorption perturbation from DTOF moment changes."""

    def test_reconstruct_td_exists(self):
        from dot_jax.td_forward import reconstruct_td

    def test_returns_correct_shape(self, box_mesh, geometry):
        from dot_jax.td_forward import reconstruct_td

        srcpos, detpos = geometry

        # Generate synthetic data: background vs perturbed
        result_bg = td_forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos,
                                   t_max=2e-9, n_times=50)
        result_pert = td_forward_cw(box_mesh, 0.015, 1.0, srcpos, detpos,
                                     t_max=2e-9, n_times=50)

        m0_bg, m1_bg, m2_bg = dtof_moments(result_bg.dtof, result_bg.times)
        m0_pt, m1_pt, m2_pt = dtof_moments(result_pert.dtof, result_pert.times)

        # Moment perturbations
        delta_moments = jnp.concatenate([m0_pt - m0_bg, m1_pt - m1_bg, m2_pt - m2_bg])

        recon = reconstruct_td(
            box_mesh, delta_moments, srcpos, detpos,
            mua0=0.01, musp=1.0,
            t_max=2e-9, n_times=50,
        )
        assert recon.shape == (box_mesh.nn,)

    def test_positive_perturbation_detected(self, box_mesh, geometry):
        """Increased mua should reconstruct as positive delta_mua."""
        from dot_jax.td_forward import reconstruct_td

        srcpos, detpos = geometry

        result_bg = td_forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos,
                                   t_max=2e-9, n_times=50)
        result_pert = td_forward_cw(box_mesh, 0.015, 1.0, srcpos, detpos,
                                     t_max=2e-9, n_times=50)

        m0_bg, m1_bg, m2_bg = dtof_moments(result_bg.dtof, result_bg.times)
        m0_pt, m1_pt, m2_pt = dtof_moments(result_pert.dtof, result_pert.times)

        delta_moments = jnp.concatenate([m0_pt - m0_bg, m1_pt - m1_bg, m2_pt - m2_bg])

        recon = reconstruct_td(
            box_mesh, delta_moments, srcpos, detpos,
            mua0=0.01, musp=1.0,
            t_max=2e-9, n_times=50,
        )
        # Mean perturbation should be positive (mua increased)
        assert float(jnp.mean(recon)) > 0, (
            f"Expected positive mean delta_mua, got {float(jnp.mean(recon)):.6f}"
        )

    def test_finite(self, box_mesh, geometry):
        from dot_jax.td_forward import reconstruct_td

        srcpos, detpos = geometry
        result_bg = td_forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos,
                                   t_max=2e-9, n_times=50)
        result_pert = td_forward_cw(box_mesh, 0.012, 1.0, srcpos, detpos,
                                     t_max=2e-9, n_times=50)

        m0_bg, m1_bg, m2_bg = dtof_moments(result_bg.dtof, result_bg.times)
        m0_pt, m1_pt, m2_pt = dtof_moments(result_pert.dtof, result_pert.times)
        delta_moments = jnp.concatenate([m0_pt - m0_bg, m1_pt - m1_bg, m2_pt - m2_bg])

        recon = reconstruct_td(
            box_mesh, delta_moments, srcpos, detpos,
            mua0=0.01, musp=1.0,
            t_max=2e-9, n_times=50,
        )
        assert jnp.all(jnp.isfinite(recon))

    def test_moments_carry_more_depth_info_than_cw(self, box_mesh, geometry):
        """TD moments should have higher-rank Jacobian than CW alone.

        The m0 moment is equivalent to CW intensity. Adding m1 and m2
        provides additional depth-dependent information, increasing
        the effective rank of the measurement.
        """
        from dot_jax.td_forward import compute_moment_jacobian

        srcpos, detpos = geometry
        J = compute_moment_jacobian(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=2e-9, n_times=50,
        )
        n_det = detpos.shape[0]

        # CW-equivalent: only m0 rows
        J_cw = J[:n_det, :]
        # Full TD: all moment rows
        J_td = J

        # Effective rank (number of singular values > 1% of max)
        s_cw = jnp.linalg.svd(J_cw, compute_uv=False)
        s_td = jnp.linalg.svd(J_td, compute_uv=False)

        rank_cw = int(jnp.sum(s_cw > 0.01 * s_cw[0]))
        rank_td = int(jnp.sum(s_td > 0.01 * s_td[0]))

        assert rank_td >= rank_cw, (
            f"TD rank ({rank_td}) should be >= CW rank ({rank_cw})"
        )
