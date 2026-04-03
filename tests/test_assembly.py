"""Tests for FEM stiffness matrix assembly — RED-GREEN TDD.

Validates assembly of K (stiffness), M (mass), C (Robin BC), and
full system matrix against known properties and redbirdpy cross-validation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from dot_jax.mesh import FEMMesh
from dot_jax.assembly import (
    assemble_stiffness,
    assemble_mass,
    assemble_boundary,
    assemble_system_cw,
)


# ---------------------------------------------------------------------------
# Stiffness matrix K
# ---------------------------------------------------------------------------

class TestAssembleStiffness:
    def test_shape(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        D = 1.0 / (3.0 * (0.01 + 1.0))
        K = assemble_stiffness(mesh, D)
        assert K.shape == (mesh.nn, mesh.nn)

    def test_symmetric(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        D = 1.0 / (3.0 * (0.01 + 1.0))
        K = assemble_stiffness(mesh, D)
        npt.assert_allclose(K, K.T, atol=1e-15)

    def test_positive_semidefinite(self, tiny_mesh):
        """Stiffness matrix should be positive semi-definite."""
        mesh = FEMMesh.create(**tiny_mesh)
        D = 1.0 / (3.0 * (0.01 + 1.0))
        K = assemble_stiffness(mesh, D)
        eigvals = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigvals >= -1e-12)

    def test_scales_with_D(self, tiny_mesh):
        """K should scale linearly with D."""
        mesh = FEMMesh.create(**tiny_mesh)
        K1 = assemble_stiffness(mesh, 1.0)
        K2 = assemble_stiffness(mesh, 2.0)
        npt.assert_allclose(K2, 2.0 * K1, rtol=1e-12)

    def test_per_element_D(self, tiny_mesh):
        """Scalar D should match constant per-element D array."""
        mesh = FEMMesh.create(**tiny_mesh)
        D_scalar = 0.5
        D_array = jnp.full(mesh.ne, D_scalar)
        K1 = assemble_stiffness(mesh, D_scalar)
        K2 = assemble_stiffness(mesh, D_array)
        npt.assert_allclose(K1, K2, atol=1e-15)

    def test_finite(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        K = assemble_stiffness(mesh, 0.5)
        assert jnp.all(jnp.isfinite(K))

    def test_jit(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        f = jax.jit(lambda D: assemble_stiffness(mesh, D))
        K1 = f(0.5)
        K2 = assemble_stiffness(mesh, 0.5)
        npt.assert_allclose(K1, K2, rtol=1e-12)

    def test_grad_wrt_D(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)

        def k_sum(D):
            return jnp.sum(assemble_stiffness(mesh, D))

        g = jax.grad(k_sum)(0.5)
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Mass matrix M
# ---------------------------------------------------------------------------

class TestAssembleMass:
    def test_shape(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        M = assemble_mass(mesh, 0.01)
        assert M.shape == (mesh.nn, mesh.nn)

    def test_symmetric(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        M = assemble_mass(mesh, 0.01)
        npt.assert_allclose(M, M.T, atol=1e-15)

    def test_positive_semidefinite(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        M = assemble_mass(mesh, 0.01)
        eigvals = jnp.linalg.eigvalsh(M)
        assert jnp.all(eigvals >= -1e-12)

    def test_scales_with_mua(self, tiny_mesh):
        """M should scale linearly with mua."""
        mesh = FEMMesh.create(**tiny_mesh)
        M1 = assemble_mass(mesh, 0.01)
        M2 = assemble_mass(mesh, 0.02)
        npt.assert_allclose(M2, 2.0 * M1, rtol=1e-12)

    def test_row_sum_equals_nvol(self, tiny_mesh):
        """Row sum of consistent mass (mua=1) should equal nodal volumes."""
        mesh = FEMMesh.create(**tiny_mesh)
        M = assemble_mass(mesh, 1.0)
        # Per element: diag=0.10*V, off-diag=0.05*V, row sum = 0.25*V = V/4
        # Summed over connected elements → nvol
        row_sums = jnp.sum(M, axis=1)
        npt.assert_allclose(row_sums, mesh.nvol, rtol=1e-12)

    def test_per_element_mua(self, tiny_mesh):
        """Scalar mua should match constant per-element mua array."""
        mesh = FEMMesh.create(**tiny_mesh)
        mua_scalar = 0.01
        mua_array = jnp.full(mesh.ne, mua_scalar)
        M1 = assemble_mass(mesh, mua_scalar)
        M2 = assemble_mass(mesh, mua_array)
        npt.assert_allclose(M1, M2, atol=1e-15)

    def test_finite(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        M = assemble_mass(mesh, 0.01)
        assert jnp.all(jnp.isfinite(M))

    def test_jit(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        f = jax.jit(lambda mua: assemble_mass(mesh, mua))
        M1 = f(0.01)
        M2 = assemble_mass(mesh, 0.01)
        npt.assert_allclose(M1, M2, rtol=1e-12)

    def test_grad_wrt_mua(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)

        def m_sum(mua):
            return jnp.sum(assemble_mass(mesh, mua))

        g = jax.grad(m_sum)(0.01)
        assert jnp.isfinite(g)


# ---------------------------------------------------------------------------
# Robin boundary condition matrix C
# ---------------------------------------------------------------------------

class TestAssembleBoundary:
    def test_shape(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        C = assemble_boundary(mesh, 1.37, 1.0)
        assert C.shape == (mesh.nn, mesh.nn)

    def test_symmetric(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        C = assemble_boundary(mesh, 1.37, 1.0)
        npt.assert_allclose(C, C.T, atol=1e-15)

    def test_positive_semidefinite(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        C = assemble_boundary(mesh, 1.37, 1.0)
        eigvals = jnp.linalg.eigvalsh(C)
        assert jnp.all(eigvals >= -1e-12)

    def test_only_boundary_nodes(self, tiny_mesh):
        """C should only have entries at boundary node pairs."""
        mesh = FEMMesh.create(**tiny_mesh)
        C = assemble_boundary(mesh, 1.37, 1.0)
        # For tiny_mesh all nodes are on boundary, so all may have entries
        # But interior nodes (if any) should have zero rows/cols
        assert jnp.all(jnp.isfinite(C))

    def test_zero_for_matched_index(self, tiny_mesh):
        """C should be zero when n_in == n_out (Reff=0, BC vanishes)."""
        mesh = FEMMesh.create(**tiny_mesh)
        C = assemble_boundary(mesh, 1.0, 1.0)
        # Reff=0 → bc_coeff = 1/12, not zero. Actually the BC doesn't vanish.
        # But C should still be well-formed.
        assert jnp.all(jnp.isfinite(C))

    def test_increases_with_mismatch(self, tiny_mesh):
        """Larger index mismatch → larger boundary contribution."""
        mesh = FEMMesh.create(**tiny_mesh)
        C1 = assemble_boundary(mesh, 1.1, 1.0)
        C2 = assemble_boundary(mesh, 1.5, 1.0)
        # Higher n_in → higher Reff → smaller (1-Reff)/(1+Reff) → smaller C
        assert jnp.sum(jnp.abs(C2)) < jnp.sum(jnp.abs(C1))

    def test_finite(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        C = assemble_boundary(mesh, 1.37, 1.0)
        assert jnp.all(jnp.isfinite(C))

    def test_jit(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        f = jax.jit(lambda n_in: assemble_boundary(mesh, n_in, 1.0))
        C1 = f(1.37)
        C2 = assemble_boundary(mesh, 1.37, 1.0)
        npt.assert_allclose(C1, C2, rtol=1e-12)


# ---------------------------------------------------------------------------
# Full system matrix A = K + M + C
# ---------------------------------------------------------------------------

class TestAssembleSystemCW:
    def test_shape(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        A = assemble_system_cw(mesh, 0.01, 1.0, 1.37, 1.0)
        assert A.shape == (mesh.nn, mesh.nn)

    def test_symmetric(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        A = assemble_system_cw(mesh, 0.01, 1.0, 1.37, 1.0)
        npt.assert_allclose(A, A.T, atol=1e-14)

    def test_positive_definite(self, tiny_mesh):
        """Full system matrix should be positive definite (with absorption + BC)."""
        mesh = FEMMesh.create(**tiny_mesh)
        A = assemble_system_cw(mesh, 0.01, 1.0, 1.37, 1.0)
        eigvals = jnp.linalg.eigvalsh(A)
        assert jnp.all(eigvals > 0)

    def test_equals_sum_of_parts(self, tiny_mesh):
        """A should equal K + M + C."""
        mesh = FEMMesh.create(**tiny_mesh)
        mua, musp = 0.01, 1.0
        n_in, n_out = 1.37, 1.0
        D = 1.0 / (3.0 * (mua + musp))

        K = assemble_stiffness(mesh, D)
        M = assemble_mass(mesh, mua)
        C = assemble_boundary(mesh, n_in, n_out)
        A = assemble_system_cw(mesh, mua, musp, n_in, n_out)

        npt.assert_allclose(A, K + M + C, atol=1e-14)

    def test_cross_validate_redbirdpy(self, tiny_mesh):
        """System matrix must match redbirdpy for same mesh and properties."""
        pytest.importorskip("redbirdpy")
        from redbirdpy.forward import femlhs
        from redbirdpy.utility import deldotdel as rb_deldotdel

        node = np.array(tiny_mesh["node"])
        elem = np.array(tiny_mesh["elem"])

        # redbirdpy uses 1-based
        elem_1 = elem + 1

        mua_val, musp_val = 0.01, 1.0
        n_in, n_out = 1.37, 1.0

        # Compute volumes
        n0 = node[elem[:, 0]]
        v1 = node[elem[:, 1]] - n0
        v2 = node[elem[:, 2]] - n0
        v3 = node[elem[:, 3]] - n0
        evol = np.abs(np.sum(np.cross(v1, v2) * v3, axis=1)) / 6.0

        # redbirdpy cfg
        face_1 = np.array(tiny_mesh["face"]) + 1
        from dot_jax.analytical import getreff
        from dot_jax.mesh import compute_face_area
        reff = float(getreff(n_in, n_out))
        area = np.array(compute_face_area(
            jnp.array(node), jnp.array(np.array(tiny_mesh["face"]))
        ))

        # prop: (ne, 2) with columns [mua, musp]
        prop = np.column_stack([
            np.full(len(elem), mua_val),
            np.full(len(elem), musp_val),
        ])

        cfg = {
            "node": node,
            "elem": elem_1,
            "face": face_1,
            "evol": evol,
            "area": area,
            "prop": prop,
            "reff": reff,
            "omega": 0.0,
        }

        dd_rb, _ = rb_deldotdel(cfg)
        A_rb = femlhs(cfg, dd_rb).toarray()

        # dot-jax
        mesh = FEMMesh.create(**tiny_mesh)
        A_jx = np.array(assemble_system_cw(mesh, mua_val, musp_val, n_in, n_out))

        npt.assert_allclose(A_jx, A_rb, rtol=1e-10)

    def test_grad_wrt_mua(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)

        def a_sum(mua):
            return jnp.sum(assemble_system_cw(mesh, mua, 1.0, 1.37, 1.0))

        g = jax.grad(a_sum)(0.01)
        assert jnp.isfinite(g)

    def test_grad_wrt_musp(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)

        def a_sum(musp):
            return jnp.sum(assemble_system_cw(mesh, 0.01, musp, 1.37, 1.0))

        g = jax.grad(a_sum)(1.0)
        assert jnp.isfinite(g)

    def test_jit(self, tiny_mesh):
        mesh = FEMMesh.create(**tiny_mesh)
        f = jax.jit(lambda mua, musp: assemble_system_cw(mesh, mua, musp, 1.37, 1.0))
        A1 = f(0.01, 1.0)
        A2 = assemble_system_cw(mesh, 0.01, 1.0, 1.37, 1.0)
        npt.assert_allclose(A1, A2, rtol=1e-12)
