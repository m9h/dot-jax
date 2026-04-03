#!/usr/bin/env python
"""Tutorial 2: Chromophore Spectroscopy and Optical Properties.

Demonstrates the spectroscopic basis of DOT and fNIRS: how tissue
absorption depends on chromophore concentrations via the Beer-Lambert law,
and how scattering follows a wavelength power law.

The modified Beer-Lambert law relates absorption to chromophore
concentrations:

    mua(lambda) = sum_i  epsilon_i(lambda) * c_i

where epsilon_i(lambda) is the molar extinction coefficient of
chromophore i at wavelength lambda, and c_i is its concentration.

For fNIRS, the primary chromophores are:
    - HbO2 (oxygenated hemoglobin)
    - Hb   (deoxygenated hemoglobin)
    - H2O  (water)

The ~800 nm isosbestic point (where HbO2 and Hb have equal extinction)
is a key feature exploited in fNIRS system design.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dot_jax.property import (
    extinction,
    get_chromophore_table,
    mua_from_concentrations,
    musp_from_scattering,
    musp2sasp,
)

# --- Extinction spectra ---
wavelengths = np.arange(650, 1000, 5)
E = extinction(wavelengths, ["hbo", "hbr", "water"])

print("=== Extinction Coefficients (selected wavelengths) ===")
print(f"  {'Wavelength':>10s}  {'HbO2':>12s}  {'Hb':>12s}  {'Water':>12s}")
for wv in [690, 750, 800, 830, 850]:
    idx = np.argmin(np.abs(wavelengths - wv))
    print(
        f"  {wv:>10d} nm  "
        f"{float(E[idx, 0]):>12.6e}  "
        f"{float(E[idx, 1]):>12.6e}  "
        f"{float(E[idx, 2]):>12.6e}  1/(mm*uM) or 1/mm"
    )

# --- Find the isosbestic point ---
diff = np.array(E[:, 0] - E[:, 1])
crossings = np.where(np.diff(np.sign(diff)))[0]
if len(crossings) > 0:
    iso_wv = wavelengths[crossings[0]]
    print(f"\n  Isosbestic point (HbO2 = Hb): ~{iso_wv} nm")

# --- Compute mua from physiological concentrations ---
print("\n=== Absorption from Chromophore Concentrations ===")
E_nir = extinction([690, 830], ["hbo", "hbr"])

# Typical brain tissue: ~60 uM HbO2, ~40 uM Hb
concentrations = jnp.array([60.0, 40.0])  # uM
mua = mua_from_concentrations(E_nir, concentrations)
print(f"  [HbO2] = 60 uM, [Hb] = 40 uM")
print(f"  mua at 690 nm: {float(mua[0]):.6f} 1/mm")
print(f"  mua at 830 nm: {float(mua[1]):.6f} 1/mm")
print(f"  Total Hb = {60 + 40} uM, StO2 = {60 / (60 + 40) * 100:.0f}%")

# --- Gradient: how does mua change with HbO2? ---
print("\n=== Autodiff: d(mua)/d([HbO2]) ===")
grad_c = jax.grad(lambda c: mua_from_concentrations(E_nir, c)[0])(concentrations)
print(f"  d(mua_690)/d([HbO2]) = {float(grad_c[0]):.6e} (= E_HbO2 at 690 nm)")
print(f"  d(mua_690)/d([Hb])   = {float(grad_c[1]):.6e} (= E_Hb at 690 nm)")

# --- Scattering: power-law model ---
print("\n=== Reduced Scattering (Power-Law Model) ===")
sa, sp = 10.0, 1.2  # typical tissue values
wv = jnp.array([690.0, 750.0, 800.0, 830.0, 900.0])
musp = musp_from_scattering(sa, sp, wv)
print(f"  musp = {sa} * (lambda/500)^(-{sp})")
for i, w in enumerate(wv):
    print(f"  {int(w)} nm:  musp = {float(musp[i]):.4f} 1/mm")

# --- Round-trip: fit sa/sp from two wavelengths ---
print("\n=== Scattering Decomposition (musp2sasp) ===")
musp_2wv = musp_from_scattering(sa, sp, jnp.array([690.0, 830.0]))
sa_fit, sp_fit = musp2sasp(musp_2wv, jnp.array([690.0, 830.0]))
print(f"  Input:  sa = {sa}, sp = {sp}")
print(f"  Fitted: sa = {float(sa_fit):.4f}, sp = {float(sp_fit):.4f}")
