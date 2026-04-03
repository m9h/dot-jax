"""Chromophore extinction coefficients and optical property management.

JAX-native optical property functions for DOT/fNIRS. Setup functions
(extinction, get_chromophore_table) use scipy and run outside JIT.
Core math (mua_from_concentrations, musp_from_scattering, musp2sasp)
is JIT-compatible and differentiable.

Functions:
    extinction: Molar extinction coefficients via scipy interpolation
    get_chromophore_table: Raw chromophore lookup table
    mua_from_concentrations: mua = E @ c (JIT-compatible)
    musp_from_scattering: musp = sa * λ^(-sp) (JIT-compatible)
    musp2sasp: Fit scattering amplitude/power from two wavelengths (JIT-compatible)
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate


# =============================================================================
# Setup functions (outside JIT — use scipy / numpy)
# =============================================================================


def extinction(wavelengths, chromophores, **interp_opts):
    """Get molar extinction coefficients for chromophores.

    Uses scipy interpolation on built-in tables compiled by Scott Prahl
    from https://omlc.org/spectra/hemoglobin/

    Parameters
    ----------
    wavelengths : array_like
        Wavelengths in nm (as strings or numbers).
    chromophores : str or list of str
        Chromophore names: 'hbo', 'hbr', 'water', 'lipids', 'aa3'.
    **interp_opts
        Options passed to scipy.interpolate.interp1d.

    Returns
    -------
    extin : jnp.ndarray
        Extinction coefficients, shape (n_wv, n_chrome).
        Units: 1/(mm*uM) for hemoglobin, 1/mm for water/lipids.
    """
    chrome = _get_chromophore_data()

    if isinstance(wavelengths, (list, tuple)):
        wavelengths = [float(w) if isinstance(w, str) else w for w in wavelengths]
    wavelengths = np.atleast_1d(wavelengths).astype(float)

    if isinstance(chromophores, str):
        chromophores = [chromophores]

    extin = np.zeros((len(wavelengths), len(chromophores)))

    for j, chrom in enumerate(chromophores):
        chrom_lower = chrom.lower()
        if chrom_lower not in chrome:
            raise ValueError(
                f"Unknown chromophore: {chrom}. "
                f"Available: {list(chrome.keys())}"
            )
        spectrum = chrome[chrom_lower]
        f = interpolate.interp1d(
            spectrum[:, 0],
            spectrum[:, 1],
            kind="linear",
            fill_value="extrapolate",
            **interp_opts,
        )
        extin[:, j] = f(wavelengths)

    return jnp.array(extin)


def get_chromophore_table(name):
    """Get full chromophore lookup table.

    Parameters
    ----------
    name : str
        Chromophore name ('hbo', 'hbr', 'water', 'lipids', 'aa3').

    Returns
    -------
    table : ndarray
        (N, 2) array of [wavelength_nm, extinction_coefficient].
    """
    chrome = _get_chromophore_data()
    name = name.lower()
    if name not in chrome:
        raise ValueError(
            f"Unknown chromophore: {name}. Available: {list(chrome.keys())}"
        )
    return chrome[name]


# =============================================================================
# JIT-compatible optical property functions
# =============================================================================


@jax.jit
def mua_from_concentrations(extinction_matrix, concentrations):
    """Compute absorption coefficient from chromophore concentrations.

    Parameters
    ----------
    extinction_matrix : jnp.ndarray
        Extinction coefficients, shape (n_wv, n_chrome).
    concentrations : jnp.ndarray
        Chromophore concentrations, shape (n_chrome,) or (n_nodes, n_chrome).

    Returns
    -------
    mua : jnp.ndarray
        Absorption coefficient in 1/mm.
        Shape (n_wv,) if concentrations is 1-D, else (n_nodes, n_wv).
    """
    return concentrations @ extinction_matrix.T


@jax.jit
def musp_from_scattering(scatamp, scatpow, wavelengths):
    """Compute reduced scattering coefficient from power-law model.

    musp(λ) = scatamp * (λ/500)^(-scatpow)

    Parameters
    ----------
    scatamp : float or jnp.ndarray
        Scattering amplitude (musp at 500 nm).
    scatpow : float or jnp.ndarray
        Scattering power (wavelength exponent).
    wavelengths : jnp.ndarray
        Wavelengths in nm, shape (n_wv,).

    Returns
    -------
    musp : jnp.ndarray
        Reduced scattering coefficient in 1/mm, shape (n_wv,).
    """
    return scatamp * (wavelengths / 500.0) ** (-scatpow)


@jax.jit
def musp2sasp(musp, wavelengths):
    """Fit scattering amplitude and power from musp at two wavelengths.

    Uses the relation: musp = sa * λ^(-sp)

    Parameters
    ----------
    musp : jnp.ndarray
        Reduced scattering at two wavelengths, shape (2,).
    wavelengths : jnp.ndarray
        Wavelengths in nm, shape (2,).

    Returns
    -------
    sa : float
        Scattering amplitude.
    sp : float
        Scattering power.
    """
    lam = wavelengths / 500.0
    sp = jnp.log(musp[0] / musp[1]) / jnp.log(lam[1] / lam[0])
    sa = 0.5 * (musp[0] / lam[0] ** (-sp) + musp[1] / lam[1] ** (-sp))
    return sa, sp


# =============================================================================
# Chromophore data tables
# =============================================================================


def _get_chromophore_data():
    """Get built-in chromophore extinction coefficient tables.

    Returns
    -------
    chrome : dict
        Keys: 'hbo', 'hbr', 'water', 'lipids', 'aa3'.
        Values: (N, 2) arrays of [wavelength_nm, extinction_coeff].
        Units: HbO2/Hb in 1/(mm*uM), water/lipids in 1/mm.
    """
    chrome = {}

    # Hemoglobin data from Scott Prahl / OMLC
    # Original units: cm-1/M, converted to 1/(mm*uM) via 2.303e-7
    hb_raw = np.array([
        [250, 106112, 112736],
        [260, 116376, 116296],
        [270, 136068, 122880],
        [280, 131936, 118872],
        [290, 104752, 98364],
        [300, 65972, 64440],
        [310, 63352, 59156],
        [320, 78752, 74508],
        [330, 97512, 90856],
        [340, 107884, 108472],
        [350, 106576, 122092],
        [360, 94744, 134940],
        [370, 88176, 139968],
        [380, 109564, 145232],
        [390, 167748, 167780],
        [400, 266232, 223296],
        [410, 466840, 303956],
        [420, 480360, 407560],
        [430, 246072, 528600],
        [440, 102580, 413280],
        [450, 62816, 103292],
        [460, 44480, 23388.8],
        [470, 33209.2, 16156.4],
        [480, 26629.2, 14550],
        [490, 23684.4, 16684],
        [500, 20932.8, 20862],
        [510, 20035.2, 25773.6],
        [520, 24202.4, 31589.6],
        [530, 39956.8, 39036.4],
        [540, 53236, 46592],
        [550, 43016, 53412],
        [560, 32613.2, 53788],
        [570, 44496, 45072],
        [580, 50104, 37020],
        [590, 14400.8, 28324.4],
        [600, 3200, 14677.2],
        [610, 1506, 9443.6],
        [620, 942, 6509.6],
        [630, 610, 5148.8],
        [640, 442, 4345.2],
        [650, 368, 3750.12],
        [660, 319.6, 3226.56],
        [670, 294, 2795.12],
        [680, 277.6, 2407.92],
        [690, 276, 2334.68],
        [700, 290, 1794.28],
        [710, 314, 1540.48],
        [720, 348, 1325.88],
        [730, 390, 1102.2],
        [740, 446, 1115.88],
        [750, 518, 1405.24],
        [760, 586, 1548.52],
        [770, 650, 1311.88],
        [780, 710, 1075.44],
        [790, 756, 890.8],
        [800, 816, 761.72],
        [810, 864, 717.08],
        [820, 916, 693.76],
        [830, 974, 693.04],
        [840, 1022, 692.36],
        [850, 1058, 691.32],
        [860, 1092, 694.32],
        [870, 1128, 705.84],
        [880, 1154, 726.44],
        [890, 1178, 743.6],
        [900, 1198, 761.84],
        [910, 1214, 774.56],
        [920, 1224, 777.36],
        [930, 1222, 763.84],
        [940, 1214, 693.44],
        [950, 1204, 602.24],
        [960, 1186, 525.56],
        [970, 1162, 429.32],
        [980, 1128, 359.656],
        [990, 1080, 283.22],
        [1000, 1024, 206.784],
    ])

    # 2.303 (ln→log10) * 1e-4 (cm→mm) * 1e-3 (M→mM) * 1e-3 (mM→uM)
    conversion = 2.303e-7
    chrome["hbo"] = np.column_stack([hb_raw[:, 0], hb_raw[:, 1] * conversion])
    chrome["hbr"] = np.column_stack([hb_raw[:, 0], hb_raw[:, 2] * conversion])

    # Water absorption (1/mm) — Hale & Querry 1973
    water_wv = np.array([400, 500, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
    water_mua = np.array([
        0.00058, 0.00025, 0.0023, 0.0032, 0.006,
        0.026, 0.02, 0.043, 0.068, 0.39, 0.36,
    ]) * 0.1  # cm-1 → mm-1
    chrome["water"] = np.column_stack([water_wv, water_mua])

    # Lipids absorption (1/mm) — simplified NIR approximation
    lipid_wv = np.arange(650, 1000, 10)
    lipid_mua = 0.0005 * np.ones_like(lipid_wv, dtype=float)
    chrome["lipids"] = np.column_stack([lipid_wv, lipid_mua])

    # Cytochrome c oxidase (aa3) — oxidized-reduced difference spectrum
    aa3_wv = np.arange(650, 950, 5)
    aa3_mua = 0.5 * np.exp(-((aa3_wv - 830) ** 2) / (2 * 50**2)) + 0.4
    chrome["aa3"] = np.column_stack([aa3_wv, aa3_mua])

    return chrome
