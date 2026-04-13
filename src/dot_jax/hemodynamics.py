"""Hemodynamic preprocessing for CW-DOT and fNIRS.

Signal processing chain from raw continuous-wave intensity measurements
to chromophore concentration changes (HbO, HbR):

    I(t) → OD(t) → bandpass → downsample → reconstruct → unmix → HbO/HbR

The optical density conversion follows the modified Beer-Lambert law.
Spectral unmixing inverts the extinction matrix to separate chromophore
contributions at each wavelength.

References
----------
.. [1] M. Cope and D. T. Delpy, "System for long-term measurement of
       cerebral blood and tissue oxygenation on newborn infants by near
       infra-red transillumination," Med. Biol. Eng. Comput., vol. 26,
       pp. 289-294, 1988.
.. [2] D. T. Delpy et al., "Estimation of optical pathlength through
       tissue from direct time of flight measurement," Phys. Med. Biol.,
       vol. 33, pp. 1433-1442, 1988.

Functions:
    intensity_to_od: Raw intensity to optical density change
    bandpass_filter: Temporal bandpass (scipy, outside JIT)
    downsample: Temporal averaging/decimation
    spectral_unmix: Two-wavelength inversion to HbO/HbR
    compute_channel_snr: Signal-to-noise ratio per channel
    prune_channels: Reject bad channels by SNR/CV/saturation
    detect_motion_artifacts: Flag motion spikes by derivative threshold
    correct_motion_spline: Cubic spline interpolation over artifacts
    normalize_od: Scale OD by predicted measurement magnitude
    zscore_images: Temporal z-score per node
    compute_gvtd: Global Variance of Temporal Derivatives
    identify_short_channels: Find short source-detector distance channels
    regress_short_channels: Regress out superficial signal from long channels
"""

import jax.numpy as jnp
import numpy as np


def intensity_to_od(intensity):
    """Convert raw CW intensity to optical density change.

    OD(t) = -log(I(t) / I_mean)

    where I_mean is the temporal mean per channel.

    Parameters
    ----------
    intensity : (n_time, n_channels) — raw intensity measurements.

    Returns
    -------
    od : (n_time, n_channels) — optical density change.
    """
    I_mean = jnp.mean(intensity, axis=0, keepdims=True)
    return -jnp.log(intensity / I_mean)


def bandpass_filter(data, fs, low=0.01, high=0.5, order=5):
    """Bandpass filter time series data (scipy, outside JIT).

    Removes physiological noise (cardiac, respiratory) and slow drift.
    Uses a zero-phase Butterworth filter.

    Parameters
    ----------
    data : (n_time, n_channels) — time series.
    fs : float — sampling frequency (Hz).
    low : float — low cutoff frequency (Hz).
    high : float — high cutoff frequency (Hz).
    order : int — filter order.

    Returns
    -------
    filtered : (n_time, n_channels) — bandpass-filtered data.
    """
    from scipy.signal import butter, filtfilt

    data_np = np.asarray(data)
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = filtfilt(b, a, data_np, axis=0)
    return jnp.array(filtered)


def downsample(data, factor):
    """Downsample by block averaging.

    Parameters
    ----------
    data : (n_time, n_channels) — time series.
    factor : int — downsampling factor.

    Returns
    -------
    downsampled : (n_time // factor, n_channels) — averaged data.
    """
    n_time = data.shape[0]
    n_out = n_time // factor
    # Truncate to exact multiple
    truncated = data[:n_out * factor]
    # Reshape and average
    reshaped = truncated.reshape(n_out, factor, -1)
    return jnp.mean(reshaped, axis=1)


def spectral_unmix(delta_mua, extinction_matrix):
    """Unmix absorption changes to chromophore concentration changes.

    Solves: delta_mua(lambda) = E(lambda) @ delta_c
    for delta_c = [delta_HbO, delta_HbR] at each node and time point.

    Parameters
    ----------
    delta_mua : (n_wv, n_time, n_nodes) — absorption change per wavelength.
    extinction_matrix : (n_wv, n_chromophores) — extinction coefficients.

    Returns
    -------
    delta_hbo : (n_time, n_nodes) — oxyhemoglobin concentration change.
    delta_hbr : (n_time, n_nodes) — deoxyhemoglobin concentration change.
    """
    # E is (n_wv, n_chrome), delta_mua is (n_wv, n_time, n_nodes)
    # Solve least-squares: E @ c = mua for each (time, node)
    E_inv = jnp.linalg.pinv(extinction_matrix)  # (n_chrome, n_wv)

    # delta_c = E_inv @ delta_mua: (n_chrome, n_wv) @ (n_wv, n_time, n_nodes)
    delta_c = jnp.einsum("cw,wtn->tnc", E_inv, delta_mua)

    return delta_c[:, :, 0], delta_c[:, :, 1]  # hbo, hbr


# =============================================================================
# Channel quality control
# =============================================================================


def compute_channel_snr(intensity):
    """Compute signal-to-noise ratio per channel.

    SNR = mean / std over the time axis.

    Parameters
    ----------
    intensity : (n_time, n_channels) — raw intensity.

    Returns
    -------
    snr : (n_channels,) — SNR per channel.
    """
    mu = jnp.mean(intensity, axis=0)
    sigma = jnp.std(intensity, axis=0)
    return jnp.abs(mu) / jnp.maximum(sigma, 1e-30)


def prune_channels(data, snr=None, snr_threshold=2.0, cv_threshold=0.5,
                   sat_threshold=0.999):
    """Mark good channels based on quality metrics.

    A channel is rejected if any of:
    - SNR < snr_threshold
    - Coefficient of variation (std/mean) > cv_threshold
    - Fraction of samples at min or max > sat_threshold

    Parameters
    ----------
    data : (n_time, n_channels) — intensity or OD data.
    snr : (n_channels,) optional — precomputed SNR.
    snr_threshold : float — minimum acceptable SNR.
    cv_threshold : float — maximum acceptable CV.
    sat_threshold : float — max fraction of identical samples.

    Returns
    -------
    good_mask : (n_channels,) bool — True for good channels.
    """
    data_np = np.asarray(data)
    n_time, n_ch = data_np.shape

    if snr is None:
        snr = np.asarray(compute_channel_snr(jnp.array(data_np)))

    good = np.ones(n_ch, dtype=bool)

    # SNR check
    good &= snr >= snr_threshold

    # CV check
    mu = np.mean(data_np, axis=0)
    sigma = np.std(data_np, axis=0)
    cv = np.where(np.abs(mu) > 1e-30, sigma / np.abs(mu), np.inf)
    good &= cv <= cv_threshold

    # Saturation check: fraction of samples at min or max
    for ch in range(n_ch):
        col = data_np[:, ch]
        if np.std(col) < 1e-30:
            good[ch] = False
            continue
        frac_min = np.sum(col == col.min()) / n_time
        frac_max = np.sum(col == col.max()) / n_time
        if frac_min > sat_threshold or frac_max > sat_threshold:
            good[ch] = False

    return good


# =============================================================================
# Motion artifact detection and correction
# =============================================================================


def detect_motion_artifacts(data, fs, std_threshold=5.0):
    """Detect motion artifacts by temporal derivative threshold.

    Flags samples where |d(data)/dt| > std_threshold * std(derivative).

    Parameters
    ----------
    data : (n_time, n_channels) — time series (OD or intensity).
    fs : float — sampling frequency (Hz).
    std_threshold : float — threshold in units of derivative std.

    Returns
    -------
    artifact_mask : (n_time, n_channels) bool — True at artifact samples.
    """
    data_np = np.asarray(data)
    deriv = np.diff(data_np, axis=0) * fs
    deriv_std = np.std(deriv, axis=0, keepdims=True)
    deriv_std = np.maximum(deriv_std, 1e-30)

    artifact_deriv = np.abs(deriv) > std_threshold * deriv_std

    # Expand mask: flag both the sample before and after each spike
    mask = np.zeros_like(data_np, dtype=bool)
    mask[:-1] |= artifact_deriv
    mask[1:] |= artifact_deriv

    return mask


def correct_motion_spline(data, artifact_mask, fs):
    """Correct motion artifacts via cubic spline interpolation.

    Replaces flagged samples with interpolated values from clean segments.

    Parameters
    ----------
    data : (n_time, n_channels) — time series.
    artifact_mask : (n_time, n_channels) bool — True at artifacts.
    fs : float — sampling frequency.

    Returns
    -------
    corrected : (n_time, n_channels) — artifact-corrected data.
    """
    from scipy.interpolate import CubicSpline

    data_np = np.array(np.asarray(data), copy=True)
    mask_np = np.asarray(artifact_mask)
    n_time = data_np.shape[0]
    t = np.arange(n_time) / fs

    for ch in range(data_np.shape[1]):
        bad = mask_np[:, ch]
        if not np.any(bad):
            continue
        good = ~bad
        if np.sum(good) < 4:
            continue
        cs = CubicSpline(t[good], data_np[good, ch])
        data_np[bad, ch] = cs(t[bad])

    return jnp.array(data_np)


# =============================================================================
# Normalization
# =============================================================================


def normalize_od(od, predicted):
    """Normalize optical density by predicted measurement magnitude.

    Converts absolute OD to relative OD for Rytov-normalized reconstruction.

    Parameters
    ----------
    od : (n_time, n_channels) — optical density changes.
    predicted : (n_channels,) — baseline predicted measurements per channel.

    Returns
    -------
    od_norm : (n_time, n_channels) — normalized OD.
    """
    return od / jnp.maximum(jnp.abs(predicted), 1e-30)


def zscore_images(images):
    """Temporal z-score per node.

    Parameters
    ----------
    images : (n_time, n_nodes) — reconstructed image time series.

    Returns
    -------
    z : (n_time, n_nodes) — z-scored images.
    """
    mu = jnp.mean(images, axis=0, keepdims=True)
    sigma = jnp.std(images, axis=0, keepdims=True)
    return jnp.where(sigma > 1e-30, (images - mu) / sigma, 0.0)


def compute_gvtd(data, fs):
    """Global Variance of Temporal Derivatives.

    Quality metric: spatial RMS of temporal derivative at each timepoint.

    Parameters
    ----------
    data : (n_time, n_channels) — time series.
    fs : float — sampling frequency.

    Returns
    -------
    gvtd : (n_time - 1,) — GVTD at each timepoint.
    """
    deriv = jnp.diff(data, axis=0) * fs
    return jnp.sqrt(jnp.mean(deriv ** 2, axis=1))


# =============================================================================
# Short-separation regression
# =============================================================================


def identify_short_channels(srcpos, detpos, ch_src, ch_det, max_distance=10.0):
    """Identify short source-detector distance channels.

    Parameters
    ----------
    srcpos : (n_src, 3) — source positions.
    detpos : (n_det, 3) — detector positions.
    ch_src : (n_ch,) — source index per channel (0-based).
    ch_det : (n_ch,) — detector index per channel (0-based).
    max_distance : float — maximum distance (mm) for short channels.

    Returns
    -------
    short_mask : (n_ch,) bool — True for short-separation channels.
    """
    srcpos = np.asarray(srcpos)
    detpos = np.asarray(detpos)
    ch_src = np.asarray(ch_src, dtype=int)
    ch_det = np.asarray(ch_det, dtype=int)

    src_coords = srcpos[ch_src]
    det_coords = detpos[ch_det]
    dist = np.sqrt(np.sum((src_coords - det_coords) ** 2, axis=1))

    return dist <= max_distance


def regress_short_channels(data, short_mask):
    """Regress out short-channel signal from long channels.

    Uses the mean of short channels as a systemic regressor.

    Parameters
    ----------
    data : (n_time, n_channels) — time series.
    short_mask : (n_channels,) bool — True for short channels.

    Returns
    -------
    corrected : (n_time, n_channels) — regressed data.
        Short channels are zeroed; long channels have systemic component removed.
    """
    data_np = np.array(np.asarray(data), copy=True, dtype=np.float64)
    short_mask_np = np.asarray(short_mask)

    if not np.any(short_mask_np):
        return jnp.array(data_np)

    # Mean short-channel signal as regressor
    regressor = np.mean(data_np[:, short_mask_np], axis=1)
    reg_norm = regressor - np.mean(regressor)
    reg_var = np.dot(reg_norm, reg_norm)

    if reg_var < 1e-30:
        return jnp.array(data_np)

    long_mask = ~short_mask_np
    for ch in np.where(long_mask)[0]:
        y = data_np[:, ch]
        beta = np.dot(y - np.mean(y), reg_norm) / reg_var
        data_np[:, ch] = y - beta * regressor

    # Zero out short channels
    data_np[:, short_mask_np] = 0.0

    return jnp.array(data_np)


def regress_global_mean(data):
    """Regress out the spatial mean from each channel.

    The global mean captures systemic physiology (cardiac, respiratory,
    Mayer waves) that is shared across all channels. Removing it reveals
    focal neural signals. Essential when short channels are unavailable.

    Parameters
    ----------
    data : (n_time, n_channels) — time series (OD or concentration).

    Returns
    -------
    corrected : (n_time, n_channels) — data with global mean removed.
    """
    data_np = np.array(np.asarray(data), copy=True, dtype=np.float64)
    global_mean = data_np.mean(axis=1)  # (n_time,)
    reg = global_mean - global_mean.mean()
    reg_var = np.dot(reg, reg)

    if reg_var < 1e-30:
        return jnp.array(data_np)

    for ch in range(data_np.shape[1]):
        y = data_np[:, ch]
        beta = np.dot(y - y.mean(), reg) / reg_var
        data_np[:, ch] = y - beta * global_mean

    return jnp.array(data_np)


def pca_filter(data, n_components=3):
    """Remove the top N principal components from the data.

    The leading PCs of multi-channel fNIRS data capture global
    physiological fluctuations (cardiac, respiratory, Mayer waves).
    Removing them enhances focal neural signals.

    Parameters
    ----------
    data : (n_time, n_channels) — time series.
    n_components : int — number of PCs to remove.

    Returns
    -------
    cleaned : (n_time, n_channels) — data with top PCs removed.
    """
    data_np = np.array(np.asarray(data), dtype=np.float64)
    mean = data_np.mean(axis=0, keepdims=True)
    centered = data_np - mean

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    recon = U[:, :n_components] @ np.diag(S[:n_components]) @ Vt[:n_components, :]
    cleaned = data_np - recon

    return jnp.array(cleaned)
