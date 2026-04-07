fNIRS Signal Processing: From Raw Intensity to Hemoglobin
==========================================================

This tutorial walks through the complete fNIRS signal processing
pipeline in dot-jax, from raw continuous-wave intensity measurements
to oxyhemoglobin (HbO) and deoxyhemoglobin (HbR) concentration
changes. Every step maps directly to functions in
:mod:`dot_jax.hemodynamics` and :mod:`dot_jax.property`.

.. contents:: In this tutorial
   :local:
   :depth: 2


Prerequisites
-------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np

   jax.config.update("jax_enable_x64", True)

   from dot_jax.hemodynamics import (
       intensity_to_od,
       bandpass_filter,
       downsample,
       spectral_unmix,
       compute_channel_snr,
       prune_channels,
       detect_motion_artifacts,
       correct_motion_spline,
       normalize_od,
       zscore_images,
       compute_gvtd,
       identify_short_channels,
       regress_short_channels,
   )
   from dot_jax.property import (
       extinction,
       get_chromophore_table,
       mua_from_concentrations,
       musp_from_scattering,
       musp2sasp,
   )


Pipeline overview
-----------------

The standard CW-fNIRS processing chain implemented in dot-jax is:

.. code-block:: text

   Raw intensity I(t)
     |
     v
   1. Optical density conversion           intensity_to_od
     |
     v
   2. Channel quality control              compute_channel_snr, prune_channels
     |
     v
   3. Motion artifact correction           detect_motion_artifacts, correct_motion_spline
     |
     v
   4. Short-channel regression             identify_short_channels, regress_short_channels
     |
     v
   5. Bandpass filtering                   bandpass_filter
     |
     v
   6. Downsampling (optional)              downsample
     |
     v
   7. Spectral unmixing to HbO/HbR        spectral_unmix
     |
     v
   8. Normalization & z-scoring            normalize_od, zscore_images

Each step is described below with working code and the underlying
physics.


Step 1: Intensity to optical density
-------------------------------------

The modified Beer-Lambert law relates the measured intensity
:math:`I(t)` to the absorption change in tissue. The optical density
(OD) change relative to the temporal mean is:

.. math::

   \mathrm{OD}(t) = -\ln\!\left(\frac{I(t)}{\bar{I}}\right)

where :math:`\bar{I}` is the mean intensity over the time series.
A *decrease* in intensity (more absorption) produces a *positive* OD
change. This convention follows Cope & Delpy (1988).

.. code-block:: python

   # Simulate 100 time points, 50 channels
   I = jnp.ones((100, 50)) * 1000.0
   od = intensity_to_od(I)

   # Constant intensity -> zero OD change
   assert od.shape == I.shape
   assert jnp.allclose(od, 0.0, atol=1e-12)

**Sign convention check:**

.. code-block:: python

   # Channel with mean = 1000
   I = jnp.array([[1000.0], [500.0], [1500.0]])
   od = intensity_to_od(I)

   assert od[1, 0] > 0   # below-mean intensity -> positive OD (more absorption)
   assert od[2, 0] < 0   # above-mean intensity -> negative OD (less absorption)

.. note::

   ``intensity_to_od`` is pure JAX and runs inside ``jax.jit``.
   The output is always finite as long as the input intensities are
   strictly positive.


Step 2: Channel quality control
--------------------------------

Before further processing, noisy or saturated channels should be
identified and excluded.

**Signal-to-noise ratio:**

.. code-block:: python

   # compute_channel_snr returns SNR = |mean| / std per channel
   I_clean = jnp.ones((1000, 3)) * 1000.0 + 1e-10 * jax.random.normal(
       jax.random.PRNGKey(0), (1000, 3)
   )
   snr = compute_channel_snr(I_clean)
   assert jnp.all(snr > 1e6)  # very high SNR for nearly constant signal

**Channel pruning:**

:func:`~dot_jax.hemodynamics.prune_channels` rejects channels that
fail any of three criteria:

1. SNR below threshold
2. Coefficient of variation (std/mean) above threshold
3. Saturation (fraction of identical samples exceeding threshold)

.. code-block:: python

   # Good channels all pass
   data = jnp.ones((100, 5)) * 500.0 + 0.1 * jax.random.normal(
       jax.random.PRNGKey(0), (100, 5)
   )
   mask = prune_channels(data, snr_threshold=2.0)
   assert jnp.all(mask)  # all channels are good

   # Inject a noisy channel -- it gets rejected
   data_bad = data.at[:, 2].set(
       jax.random.normal(jax.random.PRNGKey(0), (100,)) * 1000.0
   )
   mask = prune_channels(data_bad, snr_threshold=2.0)
   assert not mask[2]


Step 3: Motion artifact detection and correction
--------------------------------------------------

Head motion during fNIRS recording produces spike artifacts in the
optical signal. dot-jax detects these via the temporal derivative
threshold method and corrects them with cubic spline interpolation.

**Detection:**

:func:`~dot_jax.hemodynamics.detect_motion_artifacts` flags samples
where the absolute temporal derivative exceeds
``std_threshold * std(derivative)``.

.. code-block:: python

   # Clean sinusoidal signal -> no artifacts
   t = jnp.linspace(0, 10, 1000)
   data = jnp.sin(2 * jnp.pi * 0.1 * t)[:, None]
   artifact_mask = detect_motion_artifacts(data, fs=100.0, std_threshold=5.0)
   assert jnp.sum(artifact_mask) == 0

   # Inject a spike at sample 100
   data_spike = jnp.zeros((200, 1))
   data_spike = data_spike.at[100, 0].set(100.0)
   artifact_mask = detect_motion_artifacts(data_spike, fs=10.0, std_threshold=3.0)
   assert jnp.any(artifact_mask[99:102, 0])  # spike region flagged

**Correction:**

:func:`~dot_jax.hemodynamics.correct_motion_spline` replaces flagged
samples with values interpolated from neighbouring clean segments
using scipy's ``CubicSpline``.

.. code-block:: python

   corrected = correct_motion_spline(data_spike, artifact_mask, fs=10.0)
   # The corrected spike amplitude should be much smaller
   assert jnp.abs(corrected[100, 0]) < jnp.abs(data_spike[100, 0])


Step 4: Short-separation channel regression
---------------------------------------------

Short source-detector distance channels (typically <10 mm) measure
superficial (scalp) hemodynamics rather than cortical signals. By
regressing the short-channel signal out of the long channels, we can
remove systemic physiological noise (cardiac pulsation, Mayer waves,
respiration).

**Identify short channels:**

.. code-block:: python

   srcpos = jnp.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
   detpos = jnp.array([[5.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
   ch_src = jnp.array([0, 0, 1, 1])
   ch_det = jnp.array([0, 1, 0, 1])

   short_mask = identify_short_channels(
       srcpos, detpos, ch_src, ch_det, max_distance=10.0
   )
   # src0-det0 = 5 mm -> short
   assert short_mask[0]
   # src0-det1 = 50 mm -> long
   assert not short_mask[1]

**Regress out superficial signal:**

.. code-block:: python

   n_time = 500
   np.random.seed(42)
   systemic = np.sin(np.linspace(0, 10, n_time))
   brain = np.random.randn(n_time) * 0.1

   short_ch = systemic + np.random.randn(n_time) * 0.01
   long_ch = systemic + brain
   data = jnp.column_stack([short_ch, long_ch])
   short_mask = jnp.array([True, False])

   corrected = regress_short_channels(data, short_mask)

   # Correlation with systemic signal is reduced after regression
   corr_before = np.abs(np.corrcoef(systemic, np.array(data[:, 1]))[0, 1])
   corr_after = np.abs(np.corrcoef(systemic, np.array(corrected[:, 1]))[0, 1])
   assert corr_after < corr_before


Step 5: Bandpass filtering
---------------------------

A zero-phase Butterworth bandpass filter removes slow drift (below
~0.01 Hz) and fast physiological noise (cardiac > ~0.5 Hz), isolating
the hemodynamic frequency band of interest.

.. code-block:: python

   fs = 10.0  # 10 Hz sampling rate
   t = jnp.arange(0, 100, 1.0 / fs)

   # DC + hemodynamic signal at 0.1 Hz
   signal = 5.0 + 0.1 * jnp.sin(2 * jnp.pi * 0.1 * t)
   data = signal[:, None]

   filtered = bandpass_filter(data, fs=fs, low=0.01, high=0.5)

   # DC component removed
   assert jnp.abs(jnp.mean(filtered)) < 0.5

   # Passband signal preserved (within ~20% for edge effects)
   assert jnp.std(filtered) > 0.5 * jnp.std(data - jnp.mean(data))

.. note::

   ``bandpass_filter`` uses ``scipy.signal.filtfilt`` internally and
   therefore runs *outside* the JAX JIT boundary. The result is
   returned as a ``jnp.ndarray`` for downstream JAX operations.

Function signature:

.. code-block:: python

   bandpass_filter(data, fs, low=0.01, high=0.5, order=5)
   # data  : (n_time, n_channels)
   # fs    : sampling frequency in Hz
   # low   : low cutoff in Hz
   # high  : high cutoff in Hz
   # order : Butterworth filter order


Step 6: Downsampling
---------------------

After bandpass filtering, the data can be downsampled by block
averaging to reduce memory and computation for subsequent steps.

.. code-block:: python

   data = jnp.arange(100.0)[:, None]
   ds = downsample(data, factor=10)
   assert ds.shape == (10, 1)

   # First block mean: mean(0,1,...,9) = 4.5
   assert jnp.allclose(ds[0, 0], 4.5, atol=1e-10)


Step 7: Chromophore spectroscopy and spectral unmixing
-------------------------------------------------------

This is the core spectroscopic step: converting multi-wavelength
absorption changes to hemoglobin concentration changes.

Extinction coefficient tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dot-jax includes built-in chromophore extinction spectra compiled by
Scott Prahl (OMLC). Available chromophores:

- **hbo** -- oxyhemoglobin (HbO\ :sub:`2`)
- **hbr** -- deoxyhemoglobin (Hb)
- **water** -- tissue water
- **lipids** -- lipid content
- **aa3** -- cytochrome c oxidase

.. code-block:: python

   # Get the full HbO lookup table
   table = get_chromophore_table("hbo")
   assert table.ndim == 2
   assert table.shape[1] == 2  # [wavelength_nm, extinction_coeff]

   # Wavelengths are monotonically increasing
   assert np.all(np.diff(table[:, 0]) > 0)

   # All extinction values are non-negative
   assert np.all(table[:, 1] >= 0)

**Extinction at specific wavelengths:**

.. code-block:: python

   # Interpolated extinction at 690 and 830 nm for HbO and HbR
   E = extinction([690, 830], ["hbo", "hbr"])
   assert E.shape == (2, 2)  # (n_wavelengths, n_chromophores)
   assert jnp.all(E > 0)     # positive in NIR window

**Isosbestic point:**

At approximately 800 nm, the extinction of HbO and HbR are equal
(the *isosbestic point*). This is a fundamental spectroscopic fact
used to validate the tables:

.. code-block:: python

   wv = np.arange(750, 850, 1)
   E = extinction(wv, ["hbo", "hbr"])
   diff = E[:, 0] - E[:, 1]  # HbO - HbR
   crossover_idx = np.where(np.diff(np.sign(diff)))[0]
   crossover_wv = wv[crossover_idx[0]]
   assert 780 <= crossover_wv <= 820  # crossover near 800 nm

The modified Beer-Lambert law
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The absorption coefficient at wavelength :math:`\lambda` is a linear
combination of chromophore concentrations:

.. math::

   \mu_a(\lambda) = \sum_c \varepsilon_c(\lambda) \cdot C_c

where :math:`\varepsilon_c(\lambda)` are the molar extinction
coefficients and :math:`C_c` are chromophore concentrations (in
micromolar for hemoglobin). In matrix form:

.. math::

   \boldsymbol{\mu}_a = \mathbf{E} \, \mathbf{c}

.. code-block:: python

   # Compute mua from known concentrations
   E = extinction([690, 830], ["hbo", "hbr"])
   c = jnp.array([60.0, 40.0])  # 60 uM HbO, 40 uM HbR
   mua = mua_from_concentrations(E, c)
   assert mua.shape == (2,)     # one mua per wavelength
   assert jnp.all(mua > 0)

   # mua scales linearly with concentration
   mua_2x = mua_from_concentrations(E, 2.0 * c)
   assert jnp.allclose(mua_2x, 2.0 * mua, rtol=1e-12)

**Differentiability:** The gradient of :math:`\mu_a` with respect to
concentrations equals the extinction matrix -- this is exact because
the relationship is linear:

.. code-block:: python

   E_single = extinction([690], ["hbo", "hbr"])

   def scalar_mua(c):
       return mua_from_concentrations(E_single, c)[0]

   grad_c = jax.grad(scalar_mua)(jnp.array([60.0, 40.0]))
   # grad(mua, c) = E
   assert jnp.allclose(grad_c, E_single[0, :], rtol=1e-10)

Spectral unmixing: inverting Beer-Lambert
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given multi-wavelength absorption changes
:math:`\Delta\mu_a(\lambda)`, we recover chromophore concentration
changes by inverting the extinction matrix:

.. math::

   \Delta\mathbf{c} = \mathbf{E}^{+} \, \Delta\boldsymbol{\mu}_a

where :math:`\mathbf{E}^{+}` is the Moore-Penrose pseudoinverse.
:func:`~dot_jax.hemodynamics.spectral_unmix` implements this for
the standard two-chromophore (HbO/HbR) case.

.. code-block:: python

   E = extinction([750, 850], ["hbo", "hbr"])

   # Known concentration changes
   delta_hbo = jnp.array([[[1.0]]])  # shape (1, 1) for (time, nodes)
   delta_hbr = jnp.array([[[0.5]]])
   delta_c = jnp.concatenate([delta_hbo, delta_hbr], axis=-1)

   # Forward: compute absorption changes from concentrations
   delta_mua = jnp.einsum("wc,tnc->wtn", E, delta_c)

   # Inverse: recover concentrations
   hbo, hbr = spectral_unmix(delta_mua, E)
   assert jnp.allclose(hbo[0, 0], 1.0, rtol=1e-10)
   assert jnp.allclose(hbr[0, 0], 0.5, rtol=1e-10)

.. tip::

   For the two-wavelength, two-chromophore case the pseudoinverse is
   exact (the system is square). With more wavelengths than
   chromophores, ``spectral_unmix`` performs a least-squares fit.


Scattering model
^^^^^^^^^^^^^^^^

The wavelength dependence of reduced scattering follows the empirical
power-law (Mourant et al. 1997):

.. math::

   \mu_s'(\lambda) = a \left(\frac{\lambda}{500\,\mathrm{nm}}\right)^{-b}

where :math:`a` is the scattering amplitude (value of :math:`\mu_s'`
at 500 nm) and :math:`b` is the scattering power.

.. code-block:: python

   wv = jnp.array([600.0, 700.0, 800.0, 900.0])
   musp = musp_from_scattering(1e4, 1.2, wv)

   # musp decreases with wavelength for positive scattering power
   assert jnp.all(jnp.diff(musp) < 0)

**Round-trip fitting:**

Given :math:`\mu_s'` at two wavelengths, :func:`~dot_jax.property.musp2sasp`
recovers the amplitude and power:

.. code-block:: python

   sa_true, sp_true = 1e4, 1.2
   wv = jnp.array([690.0, 830.0])
   musp_in = musp_from_scattering(sa_true, sp_true, wv)

   sa_fit, sp_fit = musp2sasp(musp_in, wv)
   musp_recon = musp_from_scattering(sa_fit, sp_fit, wv)
   assert jnp.allclose(musp_recon, musp_in, rtol=1e-10)


Step 8: Normalization and quality metrics
------------------------------------------

**Normalize OD by predicted measurements:**

For Rytov-normalized reconstruction, the OD is scaled by the
baseline predicted measurement magnitude at each channel:

.. code-block:: python

   od = jnp.ones((10, 3))
   pred1 = jnp.array([1.0, 1.0, 1.0])
   pred2 = jnp.array([2.0, 2.0, 2.0])

   normed1 = normalize_od(od, pred1)
   normed2 = normalize_od(od, pred2)
   assert jnp.allclose(normed2, normed1 / 2.0, rtol=1e-12)

**Temporal z-scoring:**

.. code-block:: python

   images = jax.random.normal(jax.random.PRNGKey(0), (1000, 10)) * 5.0 + 3.0
   z = zscore_images(images)

   # Zero mean and unit variance per node
   assert jnp.allclose(jnp.mean(z, axis=0), 0.0, atol=0.05)
   assert jnp.allclose(jnp.std(z, axis=0), 1.0, atol=0.05)

**Global Variance of Temporal Derivatives (GVTD):**

GVTD is a data quality metric: the spatial RMS of the temporal
derivative at each time point. High GVTD values indicate motion or
other transient artifacts.

.. code-block:: python

   data = jax.random.normal(jax.random.PRNGKey(0), (100, 20))
   gvtd = compute_gvtd(data, fs=10.0)
   assert gvtd.shape == (99,)  # n_time - 1
   assert jnp.all(gvtd >= 0)

   # Constant data produces zero GVTD
   gvtd_const = compute_gvtd(jnp.ones((50, 10)), fs=10.0)
   assert jnp.allclose(gvtd_const, 0.0, atol=1e-12)


Putting it all together
------------------------

Here is the complete pipeline for a typical two-wavelength fNIRS
recording:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np

   jax.config.update("jax_enable_x64", True)

   from dot_jax.hemodynamics import (
       intensity_to_od, bandpass_filter, downsample,
       spectral_unmix, prune_channels,
       detect_motion_artifacts, correct_motion_spline,
       identify_short_channels, regress_short_channels,
   )
   from dot_jax.property import extinction

   # --- Configuration ---
   fs = 10.0           # Hz
   wavelengths = [750, 850]
   chromophores = ["hbo", "hbr"]
   E = extinction(wavelengths, chromophores)

   # --- Simulate multi-wavelength intensity data ---
   key = jax.random.PRNGKey(0)
   n_time, n_channels = 2000, 24
   I_750 = jnp.abs(jax.random.normal(key, (n_time, n_channels))) + 500.0
   I_850 = jnp.abs(jax.random.normal(key, (n_time, n_channels))) + 500.0

   # --- Per-wavelength preprocessing ---
   for I_wv in [I_750, I_850]:
       # 1. Intensity to OD
       od = intensity_to_od(I_wv)

       # 2. Channel quality
       mask = prune_channels(I_wv, snr_threshold=2.0)

       # 3. Motion correction
       artifacts = detect_motion_artifacts(od, fs=fs, std_threshold=5.0)
       od = correct_motion_spline(od, artifacts, fs=fs)

       # 4. Bandpass filter
       od = bandpass_filter(od, fs=fs, low=0.01, high=0.5)

       # 5. Downsample (optional)
       od = downsample(od, factor=2)

   # --- Spectral unmixing ---
   # Stack wavelengths: (n_wv, n_time, n_channels)
   # delta_mua = od_stack  (after MBLL, OD ≈ delta_mua * DPF * L)
   # hbo, hbr = spectral_unmix(delta_mua, E)

   print("Pipeline complete.")


API reference
--------------

- :func:`dot_jax.hemodynamics.intensity_to_od`
- :func:`dot_jax.hemodynamics.bandpass_filter`
- :func:`dot_jax.hemodynamics.downsample`
- :func:`dot_jax.hemodynamics.spectral_unmix`
- :func:`dot_jax.hemodynamics.compute_channel_snr`
- :func:`dot_jax.hemodynamics.prune_channels`
- :func:`dot_jax.hemodynamics.detect_motion_artifacts`
- :func:`dot_jax.hemodynamics.correct_motion_spline`
- :func:`dot_jax.hemodynamics.normalize_od`
- :func:`dot_jax.hemodynamics.zscore_images`
- :func:`dot_jax.hemodynamics.compute_gvtd`
- :func:`dot_jax.hemodynamics.identify_short_channels`
- :func:`dot_jax.hemodynamics.regress_short_channels`
- :func:`dot_jax.property.extinction`
- :func:`dot_jax.property.get_chromophore_table`
- :func:`dot_jax.property.mua_from_concentrations`
- :func:`dot_jax.property.musp_from_scattering`
- :func:`dot_jax.property.musp2sasp`
