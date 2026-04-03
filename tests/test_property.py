"""Tests for chromophore extinction and optical property functions — RED-GREEN TDD.

Cross-validates against redbirdpy and known spectroscopic facts.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from dot_jax.property import (
    extinction,
    get_chromophore_table,
    mua_from_concentrations,
    musp_from_scattering,
    musp2sasp,
)


# ---------------------------------------------------------------------------
# Chromophore data tables
# ---------------------------------------------------------------------------

class TestGetChromophoreTable:
    def test_known_chromophores(self):
        for name in ["hbo", "hbr", "water", "lipids", "aa3"]:
            table = get_chromophore_table(name)
            assert table.ndim == 2
            assert table.shape[1] == 2
            assert len(table) > 0

    def test_wavelengths_increasing(self):
        for name in ["hbo", "hbr", "water", "lipids", "aa3"]:
            table = get_chromophore_table(name)
            assert np.all(np.diff(table[:, 0]) > 0)

    def test_positive_extinction(self):
        for name in ["hbo", "hbr", "water", "lipids", "aa3"]:
            table = get_chromophore_table(name)
            assert np.all(table[:, 1] >= 0)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown chromophore"):
            get_chromophore_table("methane")

    def test_case_insensitive(self):
        t1 = get_chromophore_table("HbO")
        t2 = get_chromophore_table("hbo")
        npt.assert_array_equal(t1, t2)


# ---------------------------------------------------------------------------
# Extinction function
# ---------------------------------------------------------------------------

class TestExtinction:
    def test_shape_single(self):
        E = extinction([690], "hbo")
        assert E.shape == (1, 1)

    def test_shape_multi(self):
        E = extinction([690, 830], ["hbo", "hbr"])
        assert E.shape == (2, 2)

    def test_positive_nir(self):
        E = extinction([690, 830], ["hbo", "hbr"])
        assert jnp.all(E > 0)

    def test_hbo_hbr_crossover_near_800nm(self):
        """HbO2 and Hb extinction cross near ~800 nm (isosbestic point)."""
        wv = np.arange(750, 850, 1)
        E = extinction(wv, ["hbo", "hbr"])
        # Find where HbO becomes larger than HbR
        diff = E[:, 0] - E[:, 1]
        crossover_idx = np.where(np.diff(np.sign(diff)))[0]
        assert len(crossover_idx) > 0, "No HbO/HbR crossover found near 800nm"
        crossover_wv = wv[crossover_idx[0]]
        assert 780 <= crossover_wv <= 820, f"Crossover at {crossover_wv}nm, expected ~800nm"

    def test_string_wavelengths(self):
        E1 = extinction(["690", "830"], ["hbo", "hbr"])
        E2 = extinction([690, 830], ["hbo", "hbr"])
        npt.assert_allclose(E1, E2, atol=1e-15)

    def test_single_chromophore_string(self):
        E = extinction([690], "hbr")
        assert E.shape == (1, 1)

    def test_unknown_chromophore_raises(self):
        with pytest.raises(ValueError, match="Unknown chromophore"):
            extinction([690], "methane")

    def test_cross_validate_redbirdpy(self):
        """Extinction values must match redbirdpy exactly (same data tables)."""
        pytest.importorskip("redbirdpy")
        from redbirdpy.property import extinction as rb_extinction

        wv = [690, 750, 830]
        chromes = ["hbo", "hbr", "water"]

        E_dot, _ = np.array(extinction(wv, chromes)), None
        E_rb, _ = rb_extinction(wv, chromes)

        npt.assert_allclose(E_dot, E_rb, rtol=1e-10)

    def test_returns_jax_array(self):
        E = extinction([690, 830], ["hbo", "hbr"])
        assert isinstance(E, jnp.ndarray)


# ---------------------------------------------------------------------------
# mua_from_concentrations
# ---------------------------------------------------------------------------

class TestMuaFromConcentrations:
    def test_shape_scalar_concentrations(self):
        E = extinction([690, 830], ["hbo", "hbr"])
        c = jnp.array([60.0, 40.0])  # uM HbO2, uM Hb
        mua = mua_from_concentrations(E, c)
        assert mua.shape == (2,)  # one mua per wavelength

    def test_shape_nodal_concentrations(self):
        E = extinction([690, 830], ["hbo", "hbr"])
        c = jnp.ones((100, 2))  # 100 nodes, 2 chromophores
        mua = mua_from_concentrations(E, c)
        assert mua.shape == (100, 2)  # (n_nodes, n_wv)

    def test_positive_for_positive_inputs(self):
        E = extinction([690, 830], ["hbo", "hbr"])
        c = jnp.array([60.0, 40.0])
        mua = mua_from_concentrations(E, c)
        assert jnp.all(mua > 0)

    def test_zero_concentrations(self):
        E = extinction([690, 830], ["hbo", "hbr"])
        c = jnp.zeros(2)
        mua = mua_from_concentrations(E, c)
        npt.assert_allclose(mua, 0.0, atol=1e-15)

    def test_linearity(self):
        """mua should scale linearly with concentration."""
        E = extinction([690, 830], ["hbo", "hbr"])
        c = jnp.array([60.0, 40.0])
        mua1 = mua_from_concentrations(E, c)
        mua2 = mua_from_concentrations(E, 2.0 * c)
        npt.assert_allclose(mua2, 2.0 * mua1, rtol=1e-12)

    def test_grad(self):
        """Gradient of mua w.r.t. concentrations should equal extinction."""
        E = extinction([690], ["hbo", "hbr"])
        c = jnp.array([60.0, 40.0])

        def scalar_mua(c):
            return mua_from_concentrations(E, c)[0]

        grad_c = jax.grad(scalar_mua)(c)
        npt.assert_allclose(grad_c, E[0, :], rtol=1e-10)

    def test_jit(self):
        E = extinction([690, 830], ["hbo", "hbr"])
        c = jnp.array([60.0, 40.0])
        f = jax.jit(lambda c: mua_from_concentrations(E, c))
        npt.assert_allclose(f(c), mua_from_concentrations(E, c), rtol=1e-12)

    def test_vmap_over_nodes(self):
        """vmap over spatial nodes."""
        E = extinction([690, 830], ["hbo", "hbr"])
        c = jnp.ones((50, 2)) * jnp.array([60.0, 40.0])
        mua = jax.vmap(lambda ci: mua_from_concentrations(E, ci))(c)
        assert mua.shape == (50, 2)
        assert jnp.all(jnp.isfinite(mua))


# ---------------------------------------------------------------------------
# musp_from_scattering
# ---------------------------------------------------------------------------

class TestMuspFromScattering:
    def test_shape(self):
        wv = jnp.array([690.0, 830.0])
        musp = musp_from_scattering(1.0, 1.0, wv)
        assert musp.shape == (2,)

    def test_positive(self):
        wv = jnp.array([690.0, 830.0])
        musp = musp_from_scattering(1e6, 1.5, wv)
        assert jnp.all(musp > 0)

    def test_decreasing_with_wavelength(self):
        """musp should decrease with wavelength for positive scatpow."""
        wv = jnp.array([600.0, 700.0, 800.0, 900.0])
        musp = musp_from_scattering(1e4, 1.2, wv)
        assert jnp.all(jnp.diff(musp) < 0)

    def test_known_values(self):
        """musp = sa * λ^(-sp), check at λ=1nm → musp=sa."""
        sa, sp = 5.0, 0.0
        musp = musp_from_scattering(sa, sp, jnp.array([690.0]))
        npt.assert_allclose(musp[0], 5.0, rtol=1e-12)

    def test_grad_wrt_scatamp(self):
        wv = jnp.array([690.0])
        grad_sa = jax.grad(lambda sa: musp_from_scattering(sa, 1.0, wv)[0])(1e4)
        expected = (690.0 / 500.0) ** (-1.0)
        npt.assert_allclose(grad_sa, expected, rtol=1e-10)

    def test_jit(self):
        wv = jnp.array([690.0, 830.0])
        f = jax.jit(lambda sa: musp_from_scattering(sa, 1.0, wv))
        npt.assert_allclose(
            f(1e4), musp_from_scattering(1e4, 1.0, wv), rtol=1e-12
        )

    def test_vmap_over_wavelengths(self):
        wv = jnp.linspace(600, 900, 31)
        musp = jax.vmap(lambda w: musp_from_scattering(1e4, 1.2, w[None]))(wv)
        assert musp.shape == (31, 1)
        assert jnp.all(jnp.isfinite(musp))


# ---------------------------------------------------------------------------
# musp2sasp (scattering decomposition)
# ---------------------------------------------------------------------------

class TestMusp2sasp:
    def test_round_trip(self):
        """Fit sa/sp, then reconstruct musp — should match."""
        sa_true, sp_true = 1e4, 1.2
        wv = jnp.array([690.0, 830.0])
        musp_in = musp_from_scattering(sa_true, sp_true, wv)

        sa_fit, sp_fit = musp2sasp(musp_in, wv)
        musp_recon = musp_from_scattering(sa_fit, sp_fit, wv)
        npt.assert_allclose(musp_recon, musp_in, rtol=1e-10)

    def test_known_power(self):
        """If musp is constant, scatpow should be 0."""
        wv = jnp.array([690.0, 830.0])
        musp_in = jnp.array([1.0, 1.0])
        _, sp = musp2sasp(musp_in, wv)
        npt.assert_allclose(sp, 0.0, atol=1e-10)

    def test_cross_validate_redbirdpy(self):
        pytest.importorskip("redbirdpy")
        from redbirdpy.property import musp2sasp as rb_musp2sasp

        musp_in = np.array([1.2, 0.9])
        wv = np.array([690.0, 830.0])

        sa_rb, sp_rb = rb_musp2sasp(musp_in, wv)
        sa_jx, sp_jx = musp2sasp(jnp.array(musp_in), jnp.array(wv))

        npt.assert_allclose(float(sa_jx), sa_rb, rtol=1e-10)
        npt.assert_allclose(float(sp_jx), sp_rb, rtol=1e-10)

    def test_grad(self):
        """musp2sasp should be differentiable w.r.t. musp."""
        wv = jnp.array([690.0, 830.0])

        def sa_fn(musp):
            sa, _ = musp2sasp(musp, wv)
            return sa

        musp_in = jnp.array([1.2, 0.9])
        grad = jax.grad(sa_fn)(musp_in)
        assert jnp.all(jnp.isfinite(grad))

    def test_jit(self):
        wv = jnp.array([690.0, 830.0])
        musp_in = jnp.array([1.2, 0.9])
        f = jax.jit(lambda m: musp2sasp(m, wv))
        sa1, sp1 = f(musp_in)
        sa2, sp2 = musp2sasp(musp_in, wv)
        npt.assert_allclose(float(sa1), float(sa2), rtol=1e-12)
        npt.assert_allclose(float(sp1), float(sp2), rtol=1e-12)
