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
