#!/usr/bin/env python
"""Tutorial 1: Analytical Solutions for Photon Diffusion.

Demonstrates the closed-form CW fluence solutions for homogeneous media.
These are useful for validating FEM forward models and understanding the
basic physics of near-infrared light transport in tissue.

The diffusion equation for steady-state (CW) photon transport is:

    -div(D * grad(phi)) + mua * phi = S

where:
    phi  = photon fluence rate (W/mm^2)
    D    = 1/(3*(mua + musp)) is the diffusion coefficient (mm)
    mua  = absorption coefficient (1/mm)
    musp = reduced scattering coefficient (1/mm)
    S    = source term

For an isotropic point source in an infinite homogeneous medium, the
Green's function solution is:

    phi(r) = (1 / (4*pi*D)) * exp(-mu_eff * r) / r

where mu_eff = sqrt(mua / D) is the effective attenuation coefficient.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dot_jax.analytical import (
    getreff,
    getdistance,
    infinite_cw,
    semi_infinite_cw,
)

# --- Tissue optical properties (typical brain tissue at 800 nm) ---
mua = 0.01      # absorption coefficient, 1/mm
musp = 1.0      # reduced scattering coefficient, 1/mm
n_tissue = 1.37  # refractive index of tissue
n_air = 1.0      # refractive index of air

# --- Source and detector geometry ---
srcpos = jnp.array([[0.0, 0.0, 0.0]])
detpos = jnp.array([
    [10.0, 0.0, 0.0],
    [20.0, 0.0, 0.0],
    [30.0, 0.0, 0.0],
    [40.0, 0.0, 0.0],
])

# --- Infinite medium fluence ---
phi_inf = infinite_cw(mua, musp, srcpos, detpos)
print("=== Infinite Medium CW Fluence ===")
for i, d in enumerate([10, 20, 30, 40]):
    print(f"  r = {d} mm:  phi = {float(phi_inf[i, 0]):.6e} W/mm^2")

# --- Semi-infinite medium (tissue-air boundary at z=0) ---
# Sources/detectors on the surface
srcpos_semi = jnp.array([[30.0, 30.0, 0.0]])
detpos_semi = jnp.array([
    [40.0, 30.0, 0.0],
    [50.0, 30.0, 0.0],
    [60.0, 30.0, 0.0],
])

phi_semi = semi_infinite_cw(mua, musp, n_tissue, n_air, srcpos_semi, detpos_semi)
phi_inf2 = infinite_cw(mua, musp, srcpos_semi, detpos_semi)

print("\n=== Semi-Infinite vs Infinite Medium ===")
print(f"  Effective reflection coefficient: Reff = {float(getreff(n_tissue, n_air)):.4f}")
for i, d in enumerate([10, 20, 30]):
    ratio = float(phi_semi[i]) / float(phi_inf2[i, 0])
    print(f"  r = {d} mm:  semi/inf ratio = {ratio:.4f}")

# --- Differentiability: sensitivity to absorption ---
print("\n=== Autodiff: d(phi)/d(mua) ===")


def total_fluence(mua):
    return jnp.sum(infinite_cw(mua, musp, srcpos, detpos))


grad_mua = jax.grad(total_fluence)(mua)
print(f"  d(sum_phi)/d(mua) = {float(grad_mua):.6e}")
print(f"  (negative = more absorption reduces fluence)")

# --- Spherical symmetry check ---
det_3d = jnp.array([
    [10.0, 0.0, 0.0],
    [0.0, 10.0, 0.0],
    [0.0, 0.0, 10.0],
    [-10.0, 0.0, 0.0],
])
phi_3d = infinite_cw(mua, musp, srcpos, det_3d)
print("\n=== Spherical Symmetry ===")
print(f"  phi at (10,0,0):  {float(phi_3d[0, 0]):.6e}")
print(f"  phi at (0,10,0):  {float(phi_3d[1, 0]):.6e}")
print(f"  phi at (0,0,10):  {float(phi_3d[2, 0]):.6e}")
print(f"  phi at (-10,0,0): {float(phi_3d[3, 0]):.6e}")
print(f"  (all equal — isotropic Green's function)")
