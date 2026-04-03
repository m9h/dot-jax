"""Tests for atlas head mesh generation and optode registration — RED-GREEN TDD.

Validates the atlas-based pipeline for DOT forward modelling using the
MNI152 atlas and ds004569 optode geometry.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

jax.config.update("jax_enable_x64", True)

from dot_jax.atlas import (
    fetch_mni152_seg,
    generate_head_mesh,
    project_to_surface,
)
from dot_jax.mesh import FEMMesh


# ---------------------------------------------------------------------------
# Atlas segmentation
# ---------------------------------------------------------------------------

class TestFetchMni152Seg:
    def test_returns_3_layers(self):
        seg = fetch_mni152_seg()
        assert seg.ndim == 4
        assert seg.shape[3] == 3  # CSF, GM, WM

    def test_probability_range(self):
        seg = fetch_mni152_seg()
        assert seg.min() >= 0.0
        assert seg.max() <= 1.0

    def test_1mm_isotropic(self):
        seg = fetch_mni152_seg()
        # MNI152 is 197x233x189 at 1mm
        assert seg.shape[:3] == (197, 233, 189)


# ---------------------------------------------------------------------------
# Head mesh generation
# ---------------------------------------------------------------------------

class TestGenerateHeadMesh:
    @pytest.fixture(scope="class")
    def head_mesh(self):
        return generate_head_mesh(max_nodes=3000, max_elem_vol=200)

    def test_returns_fem_mesh(self, head_mesh):
        assert isinstance(head_mesh, FEMMesh)

    def test_reasonable_node_count(self, head_mesh):
        assert 500 < head_mesh.nn < 10000

    def test_positive_volumes(self, head_mesh):
        assert jnp.all(head_mesh.evol > 0)

    def test_finite_deldotdel(self, head_mesh):
        assert jnp.all(jnp.isfinite(head_mesh.deldotdel))

    def test_node_coords_in_mni_space(self, head_mesh):
        """Nodes should span roughly the MNI brain extent."""
        node = np.array(head_mesh.node)
        # MNI brain: x~[-80,80], y~[-120,80], z~[-60,90]
        assert node[:, 0].min() < -30
        assert node[:, 0].max() > 30
        assert node[:, 1].min() < -60
        assert node[:, 1].max() > 30

    def test_0_based_indexing(self, head_mesh):
        assert jnp.min(head_mesh.elem) >= 0
        assert jnp.max(head_mesh.elem) < head_mesh.nn


# ---------------------------------------------------------------------------
# Optode-to-surface projection
# ---------------------------------------------------------------------------

DS004569 = Path("/data/raw/ds004569")
SNIRF_FILE = DS004569 / "sub-01/ses-01/nirs/sub-01_ses-01_task-movie.snirf"
has_ds004569 = SNIRF_FILE.exists() and SNIRF_FILE.stat().st_size > 1000


class TestProjectToSurface:
    @pytest.fixture(scope="class")
    def head_mesh(self):
        return generate_head_mesh(max_nodes=3000, max_elem_vol=200)

    def test_projects_to_surface(self, head_mesh):
        """Points outside the mesh should be projected to nearest surface node."""
        pts = np.array([[0.0, 50.0, 200.0]])  # far above head
        projected = project_to_surface(head_mesh, pts)
        assert projected.shape == (1, 3)
        # Should be on or near the mesh surface
        node = np.array(head_mesh.node)
        dists = np.linalg.norm(node - projected[0], axis=1)
        assert dists.min() < 5.0  # within 5mm of a node

    def test_preserves_shape(self, head_mesh):
        pts = np.array([[0.0, 50.0, 100.0], [10.0, 40.0, 90.0]])
        projected = project_to_surface(head_mesh, pts)
        assert projected.shape == (2, 3)

    @pytest.mark.skipif(not has_ds004569, reason="ds004569 not downloaded")
    def test_ds004569_optodes_project(self, head_mesh):
        """Real optode positions from ds004569 should project to mesh surface."""
        from dot_jax.io import read_snirf

        sd = read_snirf(SNIRF_FILE)
        all_pos = np.vstack([sd.source_pos, sd.detector_pos])
        projected = project_to_surface(head_mesh, all_pos)
        assert projected.shape == all_pos.shape

        # All projected points should be near mesh nodes
        node = np.array(head_mesh.node)
        for i in range(len(projected)):
            dists = np.linalg.norm(node - projected[i], axis=1)
            assert dists.min() < 20.0, f"Optode {i} too far from mesh: {dists.min():.1f}mm"
