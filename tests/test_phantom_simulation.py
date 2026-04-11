"""End-to-end phantom simulation — RED-GREEN TDD.

Tests that dot-jax can reconstruct a focal absorption perturbation
in a slab phantom with HD-DOT reflectance geometry. This is the
integration test that proves the full pipeline works at realistic
scale: mesh generation → forward solve → Jacobian → reconstruction
→ localization.

The slab phantom (35×35×20 mm, ~320 nodes) approximates a tissue
block imaged in reflectance mode (sources and detectors on the same
surface), which is the standard fNIRS/HD-DOT geometry.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import Delaunay

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.forward import forward_cw
from dot_jax.spectral import compute_jacobian_mua
from dot_jax.recon import reconstruct_image


# ---------------------------------------------------------------------------
# Phantom mesh generation
# ---------------------------------------------------------------------------


def _make_slab_mesh(nx=8, ny=8, nz=5, dx=5.0):
    """Generate a slab phantom mesh via scipy Delaunay.

    Parameters
    ----------
    nx, ny, nz : int — grid points per axis.
    dx : float — grid spacing (mm).

    Returns
    -------
    FEMMesh — tetrahedral slab mesh.
    """
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dx, ny)
    z = np.linspace(0, (nz - 1) * dx, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    node = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    tri = Delaunay(node)
    return FEMMesh.create(node, tri.simplices.astype(np.int32))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def slab():
    """Slab phantom: 35×35×20 mm, ~320 nodes."""
    return _make_slab_mesh(nx=8, ny=8, nz=5, dx=5.0)


@pytest.fixture(scope="module")
def optodes(slab):
    """HD-DOT reflectance geometry: sources and detectors on top face.

    3×3 source grid interleaved with 4×4 detector grid, all on
    z = z_max.  Source-detector separation ~8.75 mm (HD-DOT range).
    """
    z_max = float(np.asarray(slab.node)[:, 2].max())

    # 3×3 sources centered in the slab
    src_x = np.linspace(8.75, 26.25, 3)
    src_y = np.linspace(8.75, 26.25, 3)
    sxx, syy = np.meshgrid(src_x, src_y)
    srcpos = np.column_stack([sxx.ravel(), syy.ravel(),
                              np.full(9, z_max)])

    # 4×4 detectors offset from sources
    det_x = np.linspace(4.375, 30.625, 4)
    det_y = np.linspace(4.375, 30.625, 4)
    dxx, dyy = np.meshgrid(det_x, det_y)
    detpos = np.column_stack([dxx.ravel(), dyy.ravel(),
                              np.full(16, z_max)])

    return jnp.array(srcpos), jnp.array(detpos)


@pytest.fixture(scope="module")
def background():
    """Tissue-like optical properties."""
    return {"mua": 0.01, "musp": 1.0}  # 1/mm


@pytest.fixture(scope="module")
def jacobian(slab, optodes, background):
    """Pre-computed Jacobian for the slab phantom."""
    srcpos, detpos = optodes
    return compute_jacobian_mua(
        slab, background["mua"], background["musp"], srcpos, detpos,
    )


# ---------------------------------------------------------------------------
# Tests: mesh generation
# ---------------------------------------------------------------------------


class TestSlabMesh:
    def test_node_count(self, slab):
        """Slab mesh should have expected number of nodes."""
        assert slab.nn == 8 * 8 * 5  # 320 nodes

    def test_element_count(self, slab):
        """Delaunay tetrahedralization should produce many elements."""
        assert slab.ne >= 500

    def test_positive_volumes(self, slab):
        """All element volumes should be strictly positive."""
        assert jnp.all(slab.evol > 0)

    def test_total_volume(self, slab):
        """Total mesh volume should equal the slab dimensions."""
        expected = 35.0 * 35.0 * 20.0  # mm^3
        npt.assert_allclose(float(jnp.sum(slab.evol)), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests: forward solve at scale
# ---------------------------------------------------------------------------


class TestForwardOnSlab:
    def test_detval_finite(self, slab, optodes, background):
        """Forward solve should produce finite detector values."""
        srcpos, detpos = optodes
        result = forward_cw(slab, background["mua"], background["musp"],
                            srcpos, detpos)
        assert jnp.all(jnp.isfinite(result.detval))

    def test_nearest_pairs_positive(self, slab, optodes, background):
        """The strongest (nearest) source-detector pairs should be positive.

        On a coarse FEM mesh, far-field pairs may have small negative
        values from numerical noise — this is expected (maximum principle
        violations on linear tetrahedra).
        """
        srcpos, detpos = optodes
        result = forward_cw(slab, background["mua"], background["musp"],
                            srcpos, detpos)
        detval = result.detval
        # For each source, the max detector value should be positive
        max_per_src = jnp.max(detval, axis=0)
        assert jnp.all(max_per_src > 0), "Max detval per source must be positive"
        # Negative values should be small relative to max signal
        max_val = jnp.max(jnp.abs(detval))
        min_val = jnp.min(detval)
        assert min_val > -0.1 * max_val, (
            f"Negative values ({float(min_val):.2e}) too large "
            f"relative to max ({float(max_val):.2e})"
        )

    def test_fluence_no_large_negatives(self, slab, optodes, background):
        """Fluence should not have large negative values.

        Small negatives are expected on coarse meshes (FEM max-principle
        violation), but they should be negligible relative to the peak.
        """
        srcpos, detpos = optodes
        result = forward_cw(slab, background["mua"], background["musp"],
                            srcpos, detpos)
        max_phi = float(jnp.max(result.phi))
        min_phi = float(jnp.min(result.phi))
        assert min_phi > -0.1 * max_phi, (
            f"Min fluence ({min_phi:.2e}) too negative "
            f"relative to max ({max_phi:.2e})"
        )


# ---------------------------------------------------------------------------
# Tests: Jacobian properties at scale
# ---------------------------------------------------------------------------


class TestJacobianOnSlab:
    def test_shape(self, jacobian, slab, optodes):
        """Jacobian should be (n_meas, n_nodes)."""
        srcpos, detpos = optodes
        n_meas = srcpos.shape[0] * detpos.shape[0]
        assert jacobian.shape == (n_meas, slab.nn)

    def test_finite(self, jacobian):
        """All Jacobian entries should be finite."""
        assert jnp.all(jnp.isfinite(jacobian))

    def test_depth_sensitivity_decay(self, jacobian, slab):
        """Sensitivity should decrease with depth from the optode surface.

        This is a fundamental physical property of diffuse light:
        photons that reach deeper tissue have lower probability of
        being detected, so the Jacobian column norms should be larger
        for shallow nodes than deep nodes.
        """
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        col_norms = np.asarray(jnp.linalg.norm(jacobian, axis=0))

        # Shallow: top 40% of slab.  Deep: bottom 40%.
        shallow = node[:, 2] > 0.6 * z_max
        deep = node[:, 2] < 0.4 * z_max

        mean_shallow = col_norms[shallow].mean()
        mean_deep = col_norms[deep].mean()

        assert mean_shallow > mean_deep, (
            f"Shallow sensitivity ({mean_shallow:.2e}) should exceed "
            f"deep ({mean_deep:.2e})"
        )


# ---------------------------------------------------------------------------
# Tests: reconstruction localization (the key physics test)
# ---------------------------------------------------------------------------


class TestFocalReconstruction:
    """Validate that a focal perturbation is localized correctly.

    This is the central integration test: if we inject a Gaussian
    absorption increase at a known location and reconstruct, the peak
    of the reconstruction should be near the true perturbation center.
    """

    @staticmethod
    def _make_perturbation(node, center, sigma=5.0, amplitude=0.005):
        """Gaussian focal perturbation in mua."""
        dist = np.linalg.norm(node - center, axis=1)
        return amplitude * np.exp(-dist ** 2 / (2 * sigma ** 2))

    def test_shallow_perturbation_localized(self, slab, jacobian):
        """Shallow perturbation (5 mm deep) should be well localized.

        Target: center of slab laterally, 5 mm below the optode surface.
        Tolerance: peak within 10 mm of true center.
        """
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        target = np.array([17.5, 17.5, z_max - 5.0])

        delta_mua = self._make_perturbation(node, target)
        data = jacobian @ delta_mua

        recon = np.asarray(reconstruct_image(
            jacobian, jnp.array(data), depth_weight=True, reg_method="lcurve",
        ))

        peak_idx = int(np.argmax(recon))
        peak_pos = node[peak_idx]
        error = np.linalg.norm(peak_pos - target)

        assert error < 10.0, (
            f"Peak at {peak_pos}, target at {target}, error={error:.1f} mm"
        )

    def test_deep_perturbation_localized(self, slab, jacobian):
        """Deep perturbation (10 mm deep) should still be localized.

        Deeper perturbations are harder to reconstruct (lower sensitivity),
        so we allow a larger tolerance: 15 mm.
        """
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        target = np.array([17.5, 17.5, z_max - 10.0])

        delta_mua = self._make_perturbation(node, target)
        data = jacobian @ delta_mua

        recon = np.asarray(reconstruct_image(
            jacobian, jnp.array(data), depth_weight=True, reg_method="lcurve",
        ))

        peak_idx = int(np.argmax(recon))
        peak_pos = node[peak_idx]
        error = np.linalg.norm(peak_pos - target)

        assert error < 15.0, (
            f"Peak at {peak_pos}, target at {target}, error={error:.1f} mm"
        )

    def test_perturbation_sign_preserved(self, slab, jacobian):
        """Positive mua perturbation should reconstruct as positive peak."""
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        target = np.array([17.5, 17.5, z_max - 5.0])

        delta_mua = self._make_perturbation(node, target)
        data = jacobian @ delta_mua

        recon = reconstruct_image(
            jacobian, jnp.array(data), depth_weight=True, reg_method="lcurve",
        )

        # The peak value should be positive
        assert float(jnp.max(recon)) > 0

    def test_two_blobs_resolved(self, slab, jacobian):
        """Two separated perturbations should produce two peaks.

        Two Gaussian blobs separated by 15 mm at 5 mm depth.
        The reconstruction should have two positive local maxima
        near the true locations.
        """
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        center1 = np.array([10.0, 17.5, z_max - 5.0])
        center2 = np.array([25.0, 17.5, z_max - 5.0])

        delta_mua = (self._make_perturbation(node, center1, sigma=4.0)
                     + self._make_perturbation(node, center2, sigma=4.0))
        data = jacobian @ delta_mua

        recon = np.asarray(reconstruct_image(
            jacobian, jnp.array(data), depth_weight=True, reg_method="lcurve",
        ))

        # Find nodes near each true center (within 10 mm)
        near1 = np.linalg.norm(node - center1, axis=1) < 10.0
        near2 = np.linalg.norm(node - center2, axis=1) < 10.0

        # Both regions should have positive reconstructed values
        assert recon[near1].max() > 0, "Blob 1 region should be positive"
        assert recon[near2].max() > 0, "Blob 2 region should be positive"

        # The peak near each blob should be closer to its own center
        # than to the other center
        peak1 = node[near1][np.argmax(recon[near1])]
        peak2 = node[near2][np.argmax(recon[near2])]
        assert np.linalg.norm(peak1 - center1) < np.linalg.norm(peak1 - center2)
        assert np.linalg.norm(peak2 - center2) < np.linalg.norm(peak2 - center1)
