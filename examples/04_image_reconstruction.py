#!/usr/bin/env python
"""Tutorial 4: DOT Image Reconstruction.

Demonstrates linearised image reconstruction: recovering a spatial
map of absorption coefficient from boundary measurements.

The inverse problem in DOT is severely ill-posed: there are far more
unknowns (mua at every mesh node) than measurements (detector values
for each source-detector pair). Regularisation is essential.

This example uses:
    1. Synthetic data generated with a known mua perturbation
    2. Linearised (Born approximation) reconstruction
    3. Tikhonov regularisation with Levenberg-Marquardt damping
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.forward import forward_cw
from dot_jax.recon import reconstruct_mua

# --- Build mesh ---
node = jnp.array([
    [0.0, 0.0, 0.0],
    [20.0, 0.0, 0.0],
    [0.0, 20.0, 0.0],
    [20.0, 20.0, 0.0],
    [0.0, 0.0, 20.0],
    [20.0, 0.0, 20.0],
    [0.0, 20.0, 20.0],
    [20.0, 20.0, 20.0],
], dtype=jnp.float64)

elem = jnp.array([
    [0, 1, 2, 4],
    [1, 2, 3, 7],
    [1, 2, 4, 7],
    [1, 4, 5, 7],
    [2, 4, 6, 7],
], dtype=jnp.int32)

mesh = FEMMesh.create(node, elem)

# --- Generate synthetic data ---
srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
detpos = jnp.array([[10.0, 10.0, 10.0], [17.0, 3.0, 3.0]])
musp = 1.0
mua_background = 0.01
mua_true = 0.015  # 50% increase in absorption

print("=== Synthetic Data Generation ===")
print(f"  Background mua: {mua_background} 1/mm")
print(f"  True mua:       {mua_true} 1/mm  (+{(mua_true/mua_background - 1)*100:.0f}%)")
print(f"  Sources: {srcpos.shape[0]}, Detectors: {detpos.shape[0]}")
print(f"  Measurements: {srcpos.shape[0] * detpos.shape[0]}")

# Forward solve at true mua (this is our "measured" data)
data = forward_cw(mesh, mua_true, musp, srcpos, detpos).detval
data_bg = forward_cw(mesh, mua_background, musp, srcpos, detpos).detval

print(f"\n  Background detector values: {[f'{float(v):.4e}' for v in data_bg.ravel()]}")
print(f"  'Measured' detector values: {[f'{float(v):.4e}' for v in data.ravel()]}")
print(f"  Relative change: {[f'{float((d-b)/b)*100:.2f}%' for d, b in zip(data.ravel(), data_bg.ravel())]}")

# --- Reconstruct ---
print("\n=== Reconstruction (1 Gauss-Newton step) ===")
result = reconstruct_mua(
    mesh, data, srcpos, detpos,
    mua0=mua_background,
    musp=musp,
    max_steps=1,
    reg_param=1e-4,
)

print(f"  Residual: {float(result.residuals[0]):.6e} -> {float(result.residuals[-1]):.6e}")
print(f"  Residual reduction: {float(result.residuals[-1]/result.residuals[0])*100:.1f}%")
print(f"\n  Reconstructed mua at each node:")
for i in range(mesh.nn):
    n = mesh.node[i]
    delta = float(result.mua[i]) - mua_background
    print(f"    node {i} ({int(n[0]):>2},{int(n[1]):>2},{int(n[2]):>2}): "
          f"mua = {float(result.mua[i]):>8.5f}  "
          f"(delta = {delta:>+.5f})")

print(f"\n  Mean reconstructed mua: {float(jnp.mean(result.mua)):.5f}")
print(f"  True uniform mua:      {mua_true:.5f}")

# --- Multi-step reconstruction ---
print("\n=== Multi-Step Reconstruction ===")
result3 = reconstruct_mua(
    mesh, data, srcpos, detpos,
    mua0=mua_background,
    musp=musp,
    max_steps=3,
    reg_param=1e-4,
)
print(f"  Residuals over 3 steps:")
for i, r in enumerate(result3.residuals):
    label = "final" if i == len(result3.residuals) - 1 else f"step {i}"
    print(f"    {label}: {float(r):.6e}")
