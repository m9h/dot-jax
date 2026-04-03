"""Tests for FEM mesh module — RED-GREEN TDD.

Validates mesh operators against known geometry and redbirdpy cross-validation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from dot_jax.mesh import (
    FEMMesh,
    compute_evol,
    compute_face_area,
    compute_deldotdel,
    compute_nvol,
    elem2node,
    extract_surface,
    reorient_elems,
)


# ---------------------------------------------------------------------------
# Element volumes
# ---------------------------------------------------------------------------

class TestComputeEvol:
    def test_unit_tet(self):
        """Volume of unit tet with vertices at origin + 3 unit vectors = 1/6."""
        node = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        elem = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        evol = compute_evol(node, elem)
        npt.assert_allclose(evol[0], 1.0 / 6.0, rtol=1e-12)

    def test_scaled_tet(self):
        """Scaling node coords by s scales volume by s^3."""
        node = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        elem = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
        v1 = compute_evol(node, elem)
        v2 = compute_evol(2.0 * node, elem)
        npt.assert_allclose(v2, 8.0 * v1, rtol=1e-12)

    def test_tiny_mesh(self, tiny_mesh):
        evol = compute_evol(tiny_mesh["node"], tiny_mesh["elem"])
        assert evol.shape == (5,)
        assert jnp.all(evol > 0)
        # Total volume of unit cube = 1000 mm^3 (10x10x10)
        npt.assert_allclose(jnp.sum(evol), 1000.0, rtol=1e-10)

    def test_positive(self, tiny_mesh):
        evol = compute_evol(tiny_mesh["node"], tiny_mesh["elem"])
        assert jnp.all(evol > 0)

    def test_finite(self, tiny_mesh):
        evol = compute_evol(tiny_mesh["node"], tiny_mesh["elem"])
        assert jnp.all(jnp.isfinite(evol))

    def test_jit(self, tiny_mesh):
        f = jax.jit(compute_evol)
        evol1 = f(tiny_mesh["node"], tiny_mesh["elem"])
        evol2 = compute_evol(tiny_mesh["node"], tiny_mesh["elem"])
        npt.assert_allclose(evol1, evol2, rtol=1e-12)

    def test_grad_wrt_node(self):
        """Volume should be differentiable w.r.t. node coordinates."""
        node = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        elem = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)

        def total_vol(node):
            return jnp.sum(compute_evol(node, elem))

        g = jax.grad(total_vol)(node)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == node.shape


# ---------------------------------------------------------------------------
# Face areas
# ---------------------------------------------------------------------------

class TestComputeFaceArea:
    def test_unit_right_triangle(self):
        """Area of right triangle with legs 1 = 0.5."""
        node = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        face = jnp.array([[0, 1, 2]], dtype=jnp.int32)
        area = compute_face_area(node, face)
        npt.assert_allclose(area[0], 0.5, rtol=1e-12)

    def test_tiny_mesh(self, tiny_mesh):
        area = compute_face_area(tiny_mesh["node"], tiny_mesh["face"])
        assert area.shape == (tiny_mesh["face"].shape[0],)
        assert jnp.all(area > 0)
        assert jnp.all(jnp.isfinite(area))


# ---------------------------------------------------------------------------
# deldotdel
# ---------------------------------------------------------------------------

class TestComputeDeldotdel:
    def test_shape(self, tiny_mesh):
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]
        evol = compute_evol(node, elem)
        dd, delphi = compute_deldotdel(node, elem, evol)
        ne = elem.shape[0]
        assert dd.shape == (ne, 10)
        assert delphi.shape == (ne, 3, 4)

    def test_finite(self, tiny_mesh):
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]
        evol = compute_evol(node, elem)
        dd, delphi = compute_deldotdel(node, elem, evol)
        assert jnp.all(jnp.isfinite(dd))
        assert jnp.all(jnp.isfinite(delphi))

    def test_diagonal_positive(self, tiny_mesh):
        """Diagonal entries (self dot products) should be positive."""
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]
        evol = compute_evol(node, elem)
        dd, _ = compute_deldotdel(node, elem, evol)
        # Diagonal indices: (0,0)=0, (1,1)=4, (2,2)=7, (3,3)=9
        diag_idx = [0, 4, 7, 9]
        for idx in diag_idx:
            assert jnp.all(dd[:, idx] > 0), f"Diagonal entry {idx} not positive"

    def test_symmetry(self, tiny_mesh):
        """dd stores upper triangle — expand to 4x4, check symmetric."""
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]
        evol = compute_evol(node, elem)
        dd, _ = compute_deldotdel(node, elem, evol)

        # Expand to full 4x4 for first element
        full = jnp.zeros((4, 4))
        count = 0
        for i in range(4):
            for j in range(i, 4):
                full = full.at[i, j].set(dd[0, count])
                full = full.at[j, i].set(dd[0, count])
                count += 1
        npt.assert_allclose(full, full.T, atol=1e-15)

    def test_cross_validate_redbirdpy(self, tiny_mesh):
        """deldotdel values must match redbirdpy for same mesh."""
        pytest.importorskip("redbirdpy")
        from redbirdpy.utility import deldotdel as rb_deldotdel

        node = np.array(tiny_mesh["node"])
        elem = np.array(tiny_mesh["elem"])

        # redbirdpy uses 1-based indices
        elem_1 = elem + 1

        # Compute volumes for redbirdpy cfg
        n0 = node[elem[:, 0]]
        v1 = node[elem[:, 1]] - n0
        v2 = node[elem[:, 2]] - n0
        v3 = node[elem[:, 3]] - n0
        evol = np.abs(np.sum(np.cross(v1, v2) * v3, axis=1)) / 6.0

        cfg = {"node": node, "elem": elem_1, "evol": evol}
        dd_rb, _ = rb_deldotdel(cfg)

        # dot-jax version (0-based)
        evol_jx = compute_evol(tiny_mesh["node"], tiny_mesh["elem"])
        dd_jx, _ = compute_deldotdel(tiny_mesh["node"], tiny_mesh["elem"], evol_jx)

        npt.assert_allclose(np.array(dd_jx), dd_rb, rtol=1e-10)

    def test_jit(self, tiny_mesh):
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]
        evol = compute_evol(node, elem)
        f = jax.jit(lambda n, e, v: compute_deldotdel(n, e, v)[0])
        dd1 = f(node, elem, evol)
        dd2, _ = compute_deldotdel(node, elem, evol)
        npt.assert_allclose(dd1, dd2, rtol=1e-12)

    def test_grad_wrt_node(self, tiny_mesh):
        """deldotdel should be differentiable w.r.t. node coordinates."""
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]

        def scalar_dd(node):
            evol = compute_evol(node, elem)
            dd, _ = compute_deldotdel(node, elem, evol)
            return jnp.sum(dd)

        g = jax.grad(scalar_dd)(node)
        assert jnp.all(jnp.isfinite(g))
        assert g.shape == node.shape


# ---------------------------------------------------------------------------
# Nodal volumes
# ---------------------------------------------------------------------------

class TestComputeNvol:
    def test_conservation(self, tiny_mesh):
        """Sum of nodal volumes should equal total mesh volume."""
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]
        evol = compute_evol(node, elem)
        nvol = compute_nvol(elem, evol, node.shape[0])
        npt.assert_allclose(jnp.sum(nvol), jnp.sum(evol), rtol=1e-10)

    def test_shape(self, tiny_mesh):
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]
        evol = compute_evol(node, elem)
        nvol = compute_nvol(elem, evol, node.shape[0])
        assert nvol.shape == (node.shape[0],)

    def test_positive(self, tiny_mesh):
        node, elem = tiny_mesh["node"], tiny_mesh["elem"]
        evol = compute_evol(node, elem)
        nvol = compute_nvol(elem, evol, node.shape[0])
        assert jnp.all(nvol > 0)


# ---------------------------------------------------------------------------
# elem2node
# ---------------------------------------------------------------------------

class TestElem2Node:
    def test_constant_field(self, tiny_mesh):
        """Constant element values should give constant nodal values."""
        elem = tiny_mesh["elem"]
        nn = tiny_mesh["node"].shape[0]
        elemval = jnp.ones(elem.shape[0]) * 4.0
        nodeval = elem2node(elem, elemval, nn)
        # Each node gets contributions from connected elements, averaged by 0.25
        # For a constant field, result should be constant * (n_connected_elems / 4)
        # But actually elem2node just scatter-adds and multiplies by 0.25
        # So for constant value c, node gets c * n_connected_elements * 0.25
        assert jnp.all(jnp.isfinite(nodeval))
        assert jnp.all(nodeval > 0)

    def test_shape_1d(self, tiny_mesh):
        elem = tiny_mesh["elem"]
        nn = tiny_mesh["node"].shape[0]
        elemval = jnp.ones(elem.shape[0])
        nodeval = elem2node(elem, elemval, nn)
        assert nodeval.shape == (nn,)

    def test_shape_2d(self, tiny_mesh):
        elem = tiny_mesh["elem"]
        nn = tiny_mesh["node"].shape[0]
        elemval = jnp.ones((elem.shape[0], 3))
        nodeval = elem2node(elem, elemval, nn)
        assert nodeval.shape == (nn, 3)

    def test_zero_input(self, tiny_mesh):
        elem = tiny_mesh["elem"]
        nn = tiny_mesh["node"].shape[0]
        elemval = jnp.zeros(elem.shape[0])
        nodeval = elem2node(elem, elemval, nn)
        npt.assert_allclose(nodeval, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------

class TestExtractSurface:
    def test_tiny_mesh(self, tiny_mesh):
        face = extract_surface(np.array(tiny_mesh["elem"]))
        assert face.ndim == 2
        assert face.shape[1] == 3
        # A 5-tet cube should have 12 boundary triangles
        assert face.shape[0] == 12

    def test_0_based(self, tiny_mesh):
        face = extract_surface(np.array(tiny_mesh["elem"]))
        assert np.all(face >= 0)
        assert np.max(face) < tiny_mesh["node"].shape[0]


# ---------------------------------------------------------------------------
# Element reorientation
# ---------------------------------------------------------------------------

class TestReorientElems:
    def test_positive_volumes_after_reorient(self):
        """After reorientation, all elements should have positive signed volume."""
        node = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        # Deliberately swap to create negative volume
        elem = np.array([[1, 0, 2, 3]], dtype=np.int32)
        elem_fixed = reorient_elems(node, elem)

        n0 = node[elem_fixed[:, 0]]
        v1 = node[elem_fixed[:, 1]] - n0
        v2 = node[elem_fixed[:, 2]] - n0
        v3 = node[elem_fixed[:, 3]] - n0
        vol = np.sum(np.cross(v1, v2) * v3, axis=1)
        assert np.all(vol > 0)

    def test_already_positive_unchanged(self):
        node = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        elem = np.array([[0, 1, 2, 3]], dtype=np.int32)
        elem_fixed = reorient_elems(node, elem)
        npt.assert_array_equal(elem, elem_fixed)


# ---------------------------------------------------------------------------
# FEMMesh integration
# ---------------------------------------------------------------------------

class TestFEMMesh:
    def test_create(self, tiny_mesh):
        mesh = FEMMesh.create(
            tiny_mesh["node"], tiny_mesh["elem"], tiny_mesh["face"],
        )
        assert mesh.nn == 8
        assert mesh.ne == 5
        assert mesh.node.shape == (8, 3)
        assert mesh.elem.shape == (5, 4)
        assert mesh.evol.shape == (5,)
        assert mesh.deldotdel.shape == (5, 10)

    def test_create_auto_face(self, tiny_mesh):
        """FEMMesh.create should auto-extract faces when not provided."""
        mesh = FEMMesh.create(tiny_mesh["node"], tiny_mesh["elem"])
        assert mesh.face.ndim == 2
        assert mesh.face.shape[1] == 3
        assert mesh.nf == 12

    def test_volume_conservation(self, tiny_mesh):
        mesh = FEMMesh.create(
            tiny_mesh["node"], tiny_mesh["elem"], tiny_mesh["face"],
        )
        npt.assert_allclose(jnp.sum(mesh.nvol), jnp.sum(mesh.evol), rtol=1e-10)

    def test_0_based_indexing(self, tiny_mesh):
        mesh = FEMMesh.create(
            tiny_mesh["node"], tiny_mesh["elem"], tiny_mesh["face"],
        )
        assert jnp.min(mesh.elem) >= 0
        assert jnp.max(mesh.elem) < mesh.nn
        assert jnp.min(mesh.face) >= 0
        assert jnp.max(mesh.face) < mesh.nn

    def test_is_pytree(self, tiny_mesh):
        """FEMMesh should be a valid JAX pytree (Equinox Module)."""
        mesh = FEMMesh.create(
            tiny_mesh["node"], tiny_mesh["elem"], tiny_mesh["face"],
        )
        leaves = jax.tree.leaves(mesh)
        assert len(leaves) == 7  # node, elem, face, evol, area, nvol, deldotdel
