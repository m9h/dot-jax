"""Multi-wavelength spectral reconstruction — RED-GREEN TDD.

End-to-end test of the DOT/fNIRS pipeline: forward model at two
wavelengths (760 nm, 850 nm) → Jacobian → image reconstruction →
spectral unmixing → HbO/HbR concentration changes.

This validates the complete chain that will process real Kernel Flow 2
data: the ability to distinguish oxyhemoglobin from deoxyhemoglobin
based on their different spectral signatures.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy.spatial import Delaunay

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.property import extinction, mua_from_concentrations, musp_from_scattering
from dot_jax.spectral import compute_jacobian_mua
from dot_jax.recon import reconstruct_image
from dot_jax.hemodynamics import spectral_unmix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


WAVELENGTHS = jnp.array([760.0, 850.0])


@pytest.fixture(scope="module")
def slab():
    """Slab phantom: 35x35x20 mm."""
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
    """3x3 sources, 4x4 detectors on top face (reflectance geometry)."""
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
def ext_matrix():
    """Extinction coefficient matrix E: (2 wavelengths, 2 chromophores)."""
    return extinction(WAVELENGTHS, ["hbo", "hbr"])


@pytest.fixture(scope="module")
def tissue_background(ext_matrix):
    """Background tissue optical properties at 760 and 850 nm.

    Typical adult cortex:
      HbO = 60 uM, HbR = 40 uM, water fraction = 75%
      Scattering: sa=1.0, sp=1.0 (power-law model)
    """
    c_bg = jnp.array([60.0, 40.0])  # [HbO, HbR] in uM
    mua_bg = mua_from_concentrations(ext_matrix, c_bg)  # (2,) mua at each wv
    musp_bg = musp_from_scattering(1.0, 1.0, WAVELENGTHS)  # (2,)
    return mua_bg, musp_bg, c_bg


@pytest.fixture(scope="module")
def jacobians(slab, optodes, tissue_background):
    """Jacobian at each wavelength: list of (n_meas, nn) arrays."""
    srcpos, detpos = optodes
    mua_bg, musp_bg, _ = tissue_background
    jacs = []
    for w in range(len(WAVELENGTHS)):
        J = compute_jacobian_mua(
            slab, float(mua_bg[w]), float(musp_bg[w]), srcpos, detpos,
        )
        jacs.append(J)
    return jacs


# ---------------------------------------------------------------------------
# Helper: make a Gaussian blob perturbation in concentrations
# ---------------------------------------------------------------------------


def _concentration_blob(node, center, sigma=5.0, delta_hbo=0.0, delta_hbr=0.0):
    """Gaussian blob of chromophore concentration change.

    Returns
    -------
    delta_c : (nn, 2) — [delta_HbO, delta_HbR] per node.
    """
    dist = np.linalg.norm(node - center, axis=1)
    envelope = np.exp(-dist ** 2 / (2 * sigma ** 2))
    delta_c = np.zeros((len(node), 2))
    delta_c[:, 0] = delta_hbo * envelope
    delta_c[:, 1] = delta_hbr * envelope
    return delta_c


# ---------------------------------------------------------------------------
# Tests: extinction coefficients
# ---------------------------------------------------------------------------


class TestExtinctionSetup:
    def test_shape(self, ext_matrix):
        """E should be (2 wavelengths, 2 chromophores)."""
        assert ext_matrix.shape == (2, 2)

    def test_finite_positive(self, ext_matrix):
        """All extinction coefficients should be finite and positive."""
        assert jnp.all(jnp.isfinite(ext_matrix))
        assert jnp.all(ext_matrix > 0)

    def test_spectral_contrast(self, ext_matrix):
        """HbO and HbR should have different spectral signatures.

        At 760 nm, HbR > HbO (deoxy-dominant).
        At 850 nm, HbO > HbR (oxy-dominant).
        This crossover is what enables spectral unmixing.
        """
        hbo_760, hbr_760 = float(ext_matrix[0, 0]), float(ext_matrix[0, 1])
        hbo_850, hbr_850 = float(ext_matrix[1, 0]), float(ext_matrix[1, 1])

        assert hbr_760 > hbo_760, "HbR should dominate at 760 nm"
        assert hbo_850 > hbr_850, "HbO should dominate at 850 nm"

    def test_invertible(self, ext_matrix):
        """E must be invertible for 2-chromophore unmixing."""
        cond = float(jnp.linalg.cond(ext_matrix))
        assert cond < 100, f"Extinction matrix ill-conditioned: cond={cond:.1f}"


# ---------------------------------------------------------------------------
# Tests: multi-wavelength Jacobian
# ---------------------------------------------------------------------------


class TestMultiWavelengthJacobian:
    def test_shapes(self, jacobians, slab, optodes):
        srcpos, detpos = optodes
        n_meas = srcpos.shape[0] * detpos.shape[0]
        for J in jacobians:
            assert J.shape == (n_meas, slab.nn)

    def test_wavelength_dependence(self, jacobians):
        """Jacobians at different wavelengths should differ.

        Different mua/musp at each wavelength → different system matrix
        → different Jacobian.
        """
        assert not jnp.allclose(jacobians[0], jacobians[1])

    def test_both_finite(self, jacobians):
        for J in jacobians:
            assert jnp.all(jnp.isfinite(J))


# ---------------------------------------------------------------------------
# Tests: spectral reconstruction pipeline
# ---------------------------------------------------------------------------


class TestSpectralReconstruction:
    """End-to-end: concentration perturbation → multi-wv data → reconstruct
    delta_mua per wavelength → spectral unmix → recover HbO/HbR."""

    def test_hbo_only_perturbation(self, slab, jacobians, ext_matrix):
        """A pure HbO increase should reconstruct as HbO, not HbR.

        Inject +5 uM HbO at the slab center (5 mm deep).
        After reconstruction and unmixing, the recovered delta_HbO
        should dominate over delta_HbR.
        """
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        center = np.array([17.5, 17.5, z_max - 5.0])

        # True perturbation: +5 uM HbO, 0 HbR
        delta_c = _concentration_blob(node, center, delta_hbo=5.0, delta_hbr=0.0)

        # Convert to delta_mua at each wavelength: delta_mua = E @ delta_c
        # delta_c is (nn, 2), ext_matrix is (2, 2) → delta_mua is (nn, 2)
        delta_mua = delta_c @ ext_matrix.T  # (nn, n_wv)

        # Synthetic data at each wavelength
        recon_mua = []
        for w in range(2):
            data_w = jacobians[w] @ delta_mua[:, w]
            recon_w = reconstruct_image(
                jacobians[w], jnp.array(data_w),
                depth_weight=True, reg_method="lcurve",
            )
            recon_mua.append(np.asarray(recon_w))

        # Stack: (n_wv, n_nodes) → unmix expects (n_wv, n_time, n_nodes)
        delta_mua_recon = jnp.stack(recon_mua, axis=0)[:, None, :]  # (2, 1, nn)

        delta_hbo, delta_hbr = spectral_unmix(delta_mua_recon, ext_matrix)
        # squeeze time dim
        delta_hbo = delta_hbo[0]  # (nn,)
        delta_hbr = delta_hbr[0]  # (nn,)

        # At the perturbation center, HbO should dominate
        peak_hbo = float(jnp.max(delta_hbo))
        peak_hbr_at_hbo_peak = float(delta_hbr[jnp.argmax(delta_hbo)])

        assert peak_hbo > 0, "Recovered HbO should be positive"
        assert peak_hbo > abs(peak_hbr_at_hbo_peak), (
            f"HbO ({peak_hbo:.3f}) should dominate over HbR "
            f"({peak_hbr_at_hbo_peak:.3f}) at the perturbation peak"
        )

    def test_hbr_only_perturbation(self, slab, jacobians, ext_matrix):
        """A pure HbR increase should reconstruct as HbR, not HbO."""
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        center = np.array([17.5, 17.5, z_max - 5.0])

        delta_c = _concentration_blob(node, center, delta_hbo=0.0, delta_hbr=5.0)
        delta_mua = delta_c @ ext_matrix.T

        recon_mua = []
        for w in range(2):
            data_w = jacobians[w] @ delta_mua[:, w]
            recon_w = reconstruct_image(
                jacobians[w], jnp.array(data_w),
                depth_weight=True, reg_method="lcurve",
            )
            recon_mua.append(np.asarray(recon_w))

        delta_mua_recon = jnp.stack(recon_mua, axis=0)[:, None, :]
        delta_hbo, delta_hbr = spectral_unmix(delta_mua_recon, ext_matrix)
        delta_hbo, delta_hbr = delta_hbo[0], delta_hbr[0]

        peak_hbr = float(jnp.max(delta_hbr))
        peak_hbo_at_hbr_peak = float(delta_hbo[jnp.argmax(delta_hbr)])

        assert peak_hbr > 0, "Recovered HbR should be positive"
        assert peak_hbr > abs(peak_hbo_at_hbr_peak), (
            f"HbR ({peak_hbr:.3f}) should dominate over HbO "
            f"({peak_hbo_at_hbr_peak:.3f}) at the perturbation peak"
        )

    def test_mixed_perturbation_signs(self, slab, jacobians, ext_matrix):
        """Functional activation pattern: HbO up, HbR down.

        During cortical activation, HbO increases and HbR decreases
        (the hemodynamic response). The reconstruction should recover
        positive delta_HbO and negative delta_HbR.
        """
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        center = np.array([17.5, 17.5, z_max - 5.0])

        # Activation-like: +5 uM HbO, -2 uM HbR
        delta_c = _concentration_blob(node, center, delta_hbo=5.0, delta_hbr=-2.0)
        delta_mua = delta_c @ ext_matrix.T

        recon_mua = []
        for w in range(2):
            data_w = jacobians[w] @ delta_mua[:, w]
            recon_w = reconstruct_image(
                jacobians[w], jnp.array(data_w),
                depth_weight=True, reg_method="lcurve",
            )
            recon_mua.append(np.asarray(recon_w))

        delta_mua_recon = jnp.stack(recon_mua, axis=0)[:, None, :]
        delta_hbo, delta_hbr = spectral_unmix(delta_mua_recon, ext_matrix)
        delta_hbo, delta_hbr = delta_hbo[0], delta_hbr[0]

        # Near the perturbation center
        dist = np.linalg.norm(node - center, axis=1)
        near_center = dist < 10.0

        mean_hbo_near = float(jnp.mean(delta_hbo[near_center]))
        mean_hbr_near = float(jnp.mean(delta_hbr[near_center]))

        assert mean_hbo_near > 0, (
            f"Mean HbO near center should be positive, got {mean_hbo_near:.4f}"
        )
        assert mean_hbr_near < 0, (
            f"Mean HbR near center should be negative, got {mean_hbr_near:.4f}"
        )

    def test_hbo_localized(self, slab, jacobians, ext_matrix):
        """Recovered HbO peak should be near the true perturbation center."""
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        center = np.array([17.5, 17.5, z_max - 5.0])

        delta_c = _concentration_blob(node, center, delta_hbo=5.0, delta_hbr=0.0)
        delta_mua = delta_c @ ext_matrix.T

        recon_mua = []
        for w in range(2):
            data_w = jacobians[w] @ delta_mua[:, w]
            recon_w = reconstruct_image(
                jacobians[w], jnp.array(data_w),
                depth_weight=True, reg_method="lcurve",
            )
            recon_mua.append(np.asarray(recon_w))

        delta_mua_recon = jnp.stack(recon_mua, axis=0)[:, None, :]
        delta_hbo, _ = spectral_unmix(delta_mua_recon, ext_matrix)
        delta_hbo = np.asarray(delta_hbo[0])

        peak_pos = node[np.argmax(delta_hbo)]
        error = np.linalg.norm(peak_pos - center)

        assert error < 10.0, (
            f"HbO peak at {peak_pos}, target at {center}, error={error:.1f} mm"
        )

    def test_crosstalk_ratio(self, slab, jacobians, ext_matrix):
        """Spectral crosstalk should be bounded.

        When only HbO changes, the recovered HbR energy should be
        much smaller than the recovered HbO energy. A crosstalk ratio
        below 30% indicates the unmixing is working.
        """
        node = np.asarray(slab.node)
        z_max = node[:, 2].max()
        center = np.array([17.5, 17.5, z_max - 5.0])

        delta_c = _concentration_blob(node, center, delta_hbo=5.0, delta_hbr=0.0)
        delta_mua = delta_c @ ext_matrix.T

        recon_mua = []
        for w in range(2):
            data_w = jacobians[w] @ delta_mua[:, w]
            recon_w = reconstruct_image(
                jacobians[w], jnp.array(data_w),
                depth_weight=True, reg_method="lcurve",
            )
            recon_mua.append(np.asarray(recon_w))

        delta_mua_recon = jnp.stack(recon_mua, axis=0)[:, None, :]
        delta_hbo, delta_hbr = spectral_unmix(delta_mua_recon, ext_matrix)
        delta_hbo, delta_hbr = delta_hbo[0], delta_hbr[0]

        energy_hbo = float(jnp.sum(delta_hbo ** 2))
        energy_hbr = float(jnp.sum(delta_hbr ** 2))

        crosstalk = energy_hbr / max(energy_hbo, 1e-30)
        assert crosstalk < 0.30, (
            f"HbR-to-HbO crosstalk ratio {crosstalk:.2f} exceeds 30%"
        )
