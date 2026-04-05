"""Tests for time-domain forward model — RED-GREEN TDD.

Validates the TD diffusion equation solver: time-domain mass matrix,
ODE integration via Diffrax, DTOF computation, moment extraction,
and consistency with the CW steady-state limit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.assembly import assemble_mass, assemble_system_cw
from dot_jax.td_forward import (
    assemble_mass_time,
    td_source_pulse,
    td_forward_cw,
    dtof_moments,
    TDForwardResult,
)


@pytest.fixture
def box_mesh():
    """20x20x20 mm box mesh."""
    node = jnp.array([
        [0.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
        [20.0, 20.0, 0.0],
        [0.0, 0.0, 20.0],
        [20.0, 0.0, 20.0],
        [0.0, 20.0, 20.0],
        [20.0, 20.0, 20.0],
    ], dtype=jnp.float64)
    elem = jnp.array([
        [0, 1, 2, 4],
        [1, 2, 3, 7],
        [1, 2, 4, 7],
        [1, 4, 5, 7],
        [2, 4, 6, 7],
    ], dtype=jnp.int32)
    return FEMMesh.create(node, elem)


# ---------------------------------------------------------------------------
# Time-domain mass matrix
# ---------------------------------------------------------------------------

class TestAssembleMassTime:
    def test_shape(self, box_mesh):
        Mt = assemble_mass_time(box_mesh)
        assert Mt.shape == (box_mesh.nn, box_mesh.nn)

    def test_symmetric(self, box_mesh):
        Mt = assemble_mass_time(box_mesh)
        npt.assert_allclose(Mt, Mt.T, atol=1e-15)

    def test_positive_semidefinite(self, box_mesh):
        Mt = assemble_mass_time(box_mesh)
        eigvals = jnp.linalg.eigvalsh(Mt)
        assert jnp.all(eigvals >= -1e-12)

    def test_scales_with_n(self, box_mesh):
        """Higher refractive index → larger M_t (more optical path delay)."""
        Mt1 = assemble_mass_time(box_mesh, n=1.0)
        Mt2 = assemble_mass_time(box_mesh, n=2.0)
        npt.assert_allclose(Mt2, 2.0 * Mt1, rtol=1e-12)

    def test_proportional_to_mass(self, box_mesh):
        """M_t(n) should equal (n/c0) * consistent_mass_with_unit_coeff."""
        from dot_jax._types import C0
        n = 1.37
        Mt = assemble_mass_time(box_mesh, n=n)
        M_unit = assemble_mass(box_mesh, 1.0)  # mass matrix with mua=1
        npt.assert_allclose(Mt, (n / C0) * M_unit, rtol=1e-12)


# ---------------------------------------------------------------------------
# Source pulse
# ---------------------------------------------------------------------------

class TestTDSourcePulse:
    def test_peak_at_zero(self):
        """Gaussian pulse should peak at t=0."""
        t = jnp.linspace(-1e-9, 1e-9, 1000)
        pulse = td_source_pulse(t, fwhm=100e-12)
        peak_idx = jnp.argmax(pulse)
        assert jnp.abs(t[peak_idx]) < 5e-12

    def test_unit_area(self):
        """Pulse should integrate to approximately 1."""
        t = jnp.linspace(-1e-9, 1e-9, 10000)
        pulse = td_source_pulse(t, fwhm=100e-12)
        area = jnp.trapezoid(pulse, t)
        npt.assert_allclose(area, 1.0, rtol=0.01)

    def test_fwhm(self):
        """Width at half maximum should match specified FWHM."""
        fwhm = 100e-12
        t = jnp.linspace(-1e-9, 1e-9, 100000)
        pulse = td_source_pulse(t, fwhm=fwhm)
        half_max = jnp.max(pulse) / 2
        above_half = t[pulse >= half_max]
        measured_fwhm = above_half[-1] - above_half[0]
        npt.assert_allclose(measured_fwhm, fwhm, rtol=0.05)

    def test_positive(self):
        t = jnp.linspace(-1e-9, 1e-9, 1000)
        pulse = td_source_pulse(t, fwhm=100e-12)
        assert jnp.all(pulse >= 0)


# ---------------------------------------------------------------------------
# TD forward solve
# ---------------------------------------------------------------------------

class TestTDForwardCW:
    def test_returns_result(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        result = td_forward_cw(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=2e-9, n_times=50,
        )
        assert isinstance(result, TDForwardResult)
        assert result.dtof.ndim == 2  # (n_times, n_det)
        assert result.times.shape[0] == result.dtof.shape[0]

    def test_dtof_shape(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0], [10.0, 10.0, 10.0]])
        result = td_forward_cw(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=2e-9, n_times=100,
        )
        assert result.dtof.shape == (100, 2)

    def test_dtof_finite(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        result = td_forward_cw(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=2e-9, n_times=50,
        )
        assert jnp.all(jnp.isfinite(result.dtof))
        assert jnp.all(jnp.isfinite(result.times))

    def test_dtof_decays(self, box_mesh):
        """DTOF should rise then decay — late values smaller than peak."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        result = td_forward_cw(
            box_mesh, 0.01, 1.0, srcpos, detpos,
            t_max=5e-9, n_times=200,
        )
        dtof = result.dtof[:, 0]
        # Last quarter should be less than peak
        peak = jnp.max(dtof)
        late_mean = jnp.mean(dtof[-50:])
        assert late_mean < peak

    def test_more_absorption_faster_decay(self, box_mesh):
        """Higher mua should cause faster DTOF decay (lower 0th moment)."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        r1 = td_forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos,
                            t_max=5e-9, n_times=100)
        r2 = td_forward_cw(box_mesh, 0.05, 1.0, srcpos, detpos,
                            t_max=5e-9, n_times=100)
        # Total photon count (0th moment) should be lower with more absorption
        m0_1 = jnp.trapezoid(r1.dtof[:, 0], r1.times)
        m0_2 = jnp.trapezoid(r2.dtof[:, 0], r2.times)
        assert m0_2 < m0_1

    def test_grad_wrt_mua(self, box_mesh):
        """TD forward should be differentiable w.r.t. absorption."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])

        def total_photons(mua):
            r = td_forward_cw(box_mesh, mua, 1.0, srcpos, detpos,
                              t_max=2e-9, n_times=50)
            m0, _, _ = dtof_moments(r.dtof, r.times)
            return m0[0]

        g = jax.grad(total_photons)(0.01)
        assert jnp.isfinite(g)
        assert g < 0  # more absorption → fewer photons

    def test_grad_wrt_musp(self, box_mesh):
        """TD forward should be differentiable w.r.t. scattering."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])

        def mean_tof(musp):
            r = td_forward_cw(box_mesh, 0.01, musp, srcpos, detpos,
                              t_max=2e-9, n_times=50)
            m0, m1, _ = dtof_moments(r.dtof, r.times)
            return m1[0]

        g = jax.grad(mean_tof)(1.0)
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# DTOF moments
# ---------------------------------------------------------------------------

class TestDTOFMoments:
    def test_shape(self):
        t = jnp.linspace(0, 5e-9, 100)
        dtof = jnp.ones((100, 3))
        m0, m1, m2 = dtof_moments(dtof, t)
        assert m0.shape == (3,)
        assert m1.shape == (3,)
        assert m2.shape == (3,)

    def test_m0_positive(self):
        t = jnp.linspace(0, 5e-9, 100)
        dtof = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (100, 5)))
        m0, _, _ = dtof_moments(dtof, t)
        assert jnp.all(m0 > 0)

    def test_m1_in_time_range(self):
        """Mean ToF should be within the time window."""
        t = jnp.linspace(0, 5e-9, 100)
        # Peak at t=2ns
        dtof = jnp.exp(-0.5 * ((t - 2e-9) / 0.5e-9) ** 2)[:, None]
        m0, m1, m2 = dtof_moments(dtof, t)
        assert m1[0] > 0
        assert m1[0] < 5e-9
        # Mean should be near 2ns
        npt.assert_allclose(m1[0], 2e-9, atol=0.5e-9)

    def test_m0_is_integral(self):
        """0th moment should be the time integral of the DTOF."""
        t = jnp.linspace(0, 5e-9, 1000)
        dtof = jnp.exp(-t / 1e-9)[:, None]  # exponential decay
        m0, _, _ = dtof_moments(dtof, t)
        expected = jnp.trapezoid(dtof[:, 0], t)
        npt.assert_allclose(m0[0], expected, rtol=0.01)

    def test_finite(self):
        t = jnp.linspace(0, 5e-9, 100)
        dtof = jnp.abs(jax.random.normal(jax.random.PRNGKey(0), (100, 5))) + 1e-20
        m0, m1, m2 = dtof_moments(dtof, t)
        assert jnp.all(jnp.isfinite(m0))
        assert jnp.all(jnp.isfinite(m1))
        assert jnp.all(jnp.isfinite(m2))
