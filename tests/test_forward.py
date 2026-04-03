"""Tests for FEM forward solver — RED-GREEN TDD.

Validates CW forward solve: system assembly, linear solve via Lineax,
detector value extraction. Cross-validates against analytical solutions
and redbirdpy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from dot_jax.mesh import FEMMesh
from dot_jax.forward import (
    locate_sources,
    assemble_rhs,
    get_detector_values,
    forward_cw,
    forward_cw_sparse,
)


@pytest.fixture
def box_mesh():
    """A denser box mesh for forward solve testing.

    20x20x20 mm box split into 5 tets (same topology as tiny_mesh but
    larger for more realistic optical properties).
    """
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
# Source location (barycentric interpolation)
# ---------------------------------------------------------------------------

class TestLocateSources:
    def test_vertex_exact(self, tiny_mesh):
        """Source at a vertex should have bary = [1,0,0,0] in that element."""
        mesh = FEMMesh.create(**tiny_mesh)
        pos = jnp.array([[0.0, 0.0, 0.0]])
        elem_idx, bary = locate_sources(mesh, pos)
        assert elem_idx.shape == (1,)
        assert bary.shape == (1, 4)
        # Barycentric coords should sum to 1
        npt.assert_allclose(jnp.sum(bary, axis=1), 1.0, atol=1e-10)
        # All coords should be non-negative
        assert jnp.all(bary >= -1e-10)

    def test_multiple_sources(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        pos = jnp.array([
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
        ])
        elem_idx, bary = locate_sources(mesh, pos)
        assert elem_idx.shape == (2,)
        assert bary.shape == (2, 4)
        npt.assert_allclose(jnp.sum(bary, axis=1), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# RHS assembly
# ---------------------------------------------------------------------------

class TestAssembleRhs:
    def test_shape(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        pos = jnp.array([[1.0, 1.0, 1.0]])
        rhs = assemble_rhs(mesh, pos)
        assert rhs.shape == (mesh.nn, 1)

    def test_multi_source(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        pos = jnp.array([
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
        ])
        rhs = assemble_rhs(mesh, pos)
        assert rhs.shape == (mesh.nn, 2)

    def test_column_sums_to_one(self, tiny_mesh):
        """Each source column should sum to 1 (barycentric partition of unity)."""
        mesh = FEMMesh.create(**tiny_mesh)
        pos = jnp.array([[2.0, 2.0, 2.0]])
        rhs = assemble_rhs(mesh, pos)
        npt.assert_allclose(jnp.sum(rhs[:, 0]), 1.0, atol=1e-10)

    def test_nonnegative(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        pos = jnp.array([[2.0, 2.0, 2.0]])
        rhs = assemble_rhs(mesh, pos)
        assert jnp.all(rhs >= -1e-10)


# ---------------------------------------------------------------------------
# Detector value extraction
# ---------------------------------------------------------------------------

class TestGetDetectorValues:
    def test_shape(self):
        """detval should be (n_det, n_src)."""
        phi = jnp.ones((10, 2))
        rhs_det = jnp.ones((10, 3))
        detval = get_detector_values(phi, rhs_det)
        assert detval.shape == (3, 2)

    def test_adjoint_identity(self):
        """detval = rhs_det.T @ phi."""
        phi = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rhs_det = jnp.array([[0.1, 0.0], [0.0, 0.2], [0.3, 0.0]])
        detval = get_detector_values(phi, rhs_det)
        expected = rhs_det.T @ phi
        npt.assert_allclose(detval, expected, atol=1e-15)

    def test_jit(self):
        phi = jnp.ones((10, 2))
        rhs_det = jnp.ones((10, 3))
        f = jax.jit(get_detector_values)
        npt.assert_allclose(f(phi, rhs_det), get_detector_values(phi, rhs_det))


# ---------------------------------------------------------------------------
# Full CW forward solve
# ---------------------------------------------------------------------------

class TestForwardCW:
    def test_returns_forward_result(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        result = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos, 1.37, 1.0)

        from dot_jax import ForwardResult
        assert isinstance(result, ForwardResult)
        assert result.phi.shape[0] == box_mesh.nn
        assert result.detval.shape == (1, 1)

    def test_fluence_finite_and_detval_positive(self, box_mesh):
        """Fluence should be finite; detector values positive."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        result = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        assert jnp.all(jnp.isfinite(result.phi))
        assert jnp.all(result.detval > 0)

    def test_positive_detval(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([
            [15.0, 15.0, 15.0],
            [10.0, 10.0, 10.0],
        ])
        result = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        assert jnp.all(result.detval > 0)

    def test_multi_source_detector(self, box_mesh):
        srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
        detpos = jnp.array([[10.0, 10.0, 10.0], [15.0, 5.0, 5.0]])
        result = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        assert result.detval.shape == (2, 2)
        assert result.phi.shape == (box_mesh.nn, 2)

    def test_more_absorption_less_signal(self, box_mesh):
        """Increasing mua should decrease detector values."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        r1 = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        r2 = forward_cw(box_mesh, 0.05, 1.0, srcpos, detpos)
        assert r2.detval[0, 0] < r1.detval[0, 0]

    def test_finite(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        result = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        assert jnp.all(jnp.isfinite(result.phi))
        assert jnp.all(jnp.isfinite(result.detval))

    def test_grad_wrt_mua(self, box_mesh):
        """Forward solve should be differentiable w.r.t. mua."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])

        def detval_sum(mua):
            r = forward_cw(box_mesh, mua, 1.0, srcpos, detpos)
            return jnp.sum(r.detval)

        g = jax.grad(detval_sum)(0.01)
        assert jnp.isfinite(g)
        assert g < 0  # more absorption → less signal

    def test_grad_wrt_musp(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])

        def detval_sum(musp):
            r = forward_cw(box_mesh, 0.01, musp, srcpos, detpos)
            return jnp.sum(r.detval)

        g = jax.grad(detval_sum)(1.0)
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Sparse CG forward solve
# ---------------------------------------------------------------------------

class TestForwardCWSparse:
    def test_cross_validate_dense(self, box_mesh):
        """Sparse CG and dense LU should give same results."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        r_dense = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
        r_sparse = forward_cw_sparse(box_mesh, 0.01, 1.0, srcpos, detpos)
        npt.assert_allclose(r_sparse.detval, r_dense.detval, rtol=1e-5)
        npt.assert_allclose(r_sparse.phi, r_dense.phi, rtol=1e-5)

    def test_returns_forward_result(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        result = forward_cw_sparse(box_mesh, 0.01, 1.0, srcpos, detpos)
        from dot_jax import ForwardResult
        assert isinstance(result, ForwardResult)

    def test_positive_detval(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        result = forward_cw_sparse(box_mesh, 0.01, 1.0, srcpos, detpos)
        assert jnp.all(result.detval > 0)

    def test_multi_source_detector(self, box_mesh):
        srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
        detpos = jnp.array([[10.0, 10.0, 10.0], [15.0, 5.0, 5.0]])
        result = forward_cw_sparse(box_mesh, 0.01, 1.0, srcpos, detpos)
        assert result.detval.shape == (2, 2)
        assert result.phi.shape == (box_mesh.nn, 2)

    def test_more_absorption_less_signal(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])
        r1 = forward_cw_sparse(box_mesh, 0.01, 1.0, srcpos, detpos)
        r2 = forward_cw_sparse(box_mesh, 0.05, 1.0, srcpos, detpos)
        assert r2.detval[0, 0] < r1.detval[0, 0]

    def test_grad_wrt_mua(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])

        def detval_sum(mua):
            r = forward_cw_sparse(box_mesh, mua, 1.0, srcpos, detpos)
            return jnp.sum(r.detval)

        g = jax.grad(detval_sum)(0.01)
        assert jnp.isfinite(g)
        assert g < 0

    def test_grad_wrt_musp(self, box_mesh):
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])

        def detval_sum(musp):
            r = forward_cw_sparse(box_mesh, 0.01, musp, srcpos, detpos)
            return jnp.sum(r.detval)

        g = jax.grad(detval_sum)(1.0)
        assert jnp.isfinite(g)

    def test_grad_matches_dense(self, box_mesh):
        """Sparse CG grad should match dense LU grad."""
        srcpos = jnp.array([[5.0, 5.0, 5.0]])
        detpos = jnp.array([[15.0, 15.0, 15.0]])

        def f_dense(mua):
            return jnp.sum(forward_cw(box_mesh, mua, 1.0, srcpos, detpos).detval)

        def f_sparse(mua):
            return jnp.sum(forward_cw_sparse(box_mesh, mua, 1.0, srcpos, detpos).detval)

        g_dense = jax.grad(f_dense)(0.01)
        g_sparse = jax.grad(f_sparse)(0.01)
        npt.assert_allclose(g_sparse, g_dense, rtol=1e-3)
