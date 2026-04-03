#!/usr/bin/env python
"""Tutorial 3: FEM Forward Solve and Jacobian Computation.

Demonstrates the complete FEM-based DOT forward pipeline:
    1. Build a tetrahedral mesh
    2. Assemble the system matrix A = K + M + C
    3. Solve the linear system for photon fluence
    4. Extract detector measurements
    5. Compute the sensitivity (Jacobian) matrix

This is the core computational engine for both continuous-wave (CW)
DOT and functional near-infrared spectroscopy (fNIRS).

The system matrix encodes the physics:
    K: diffusion (how photons spread through scattering)
    M: absorption (how photons are lost to tissue)
    C: Robin BC (how photons escape at the tissue-air boundary)
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dot_jax.mesh import FEMMesh
from dot_jax.assembly import (
    assemble_stiffness,
    assemble_mass,
    assemble_boundary,
    assemble_system_cw,
)
from dot_jax.forward import forward_cw
from dot_jax.spectral import compute_jacobian_mua

# --- Build a simple box mesh ---
# 20x20x20 mm cube (5 tetrahedra, 8 nodes)
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

print("=== Mesh Summary ===")
print(f"  Nodes: {mesh.nn}, Elements: {mesh.ne}, Faces: {mesh.nf}")
print(f"  Total volume: {float(jnp.sum(mesh.evol)):.1f} mm^3")

# --- Tissue properties ---
mua = 0.01   # 1/mm
musp = 1.0   # 1/mm
n_in = 1.37
n_out = 1.0
D = 1.0 / (3.0 * (mua + musp))

# --- Inspect the system matrix components ---
K = assemble_stiffness(mesh, D)
M = assemble_mass(mesh, mua)
C = assemble_boundary(mesh, n_in, n_out)
A = assemble_system_cw(mesh, mua, musp, n_in, n_out)

print("\n=== System Matrix Components ===")
print(f"  Stiffness (K):  ||K||_F = {float(jnp.linalg.norm(K)):.6f}")
print(f"  Mass (M):       ||M||_F = {float(jnp.linalg.norm(M)):.6f}")
print(f"  Boundary (C):   ||C||_F = {float(jnp.linalg.norm(C)):.6f}")
print(f"  System (A):     ||A||_F = {float(jnp.linalg.norm(A)):.6f}")
print(f"  A is symmetric: {bool(jnp.allclose(A, A.T))}")
eigvals = jnp.linalg.eigvalsh(A)
print(f"  A eigenvalues:  min = {float(jnp.min(eigvals)):.6e}, "
      f"max = {float(jnp.max(eigvals)):.6e}")
print(f"  A is positive definite: {bool(jnp.all(eigvals > 0))}")

# --- Forward solve ---
srcpos = jnp.array([[5.0, 5.0, 5.0]])
detpos = jnp.array([
    [15.0, 15.0, 15.0],
    [15.0, 5.0, 5.0],
    [5.0, 15.0, 15.0],
])

result = forward_cw(mesh, mua, musp, srcpos, detpos, n_in, n_out)

print("\n=== Forward Solve ===")
print(f"  Source at:    (5, 5, 5) mm")
print(f"  Fluence field shape: {result.phi.shape}")
print(f"  Detector values:")
for i in range(3):
    det = detpos[i]
    print(f"    det at ({int(det[0])},{int(det[1])},{int(det[2])}): "
          f"{float(result.detval[i, 0]):.6e}")

# --- Jacobian (sensitivity matrix) ---
J = compute_jacobian_mua(mesh, mua, musp, srcpos, detpos, n_in, n_out)

print(f"\n=== Jacobian d(detval)/d(mua_node) ===")
print(f"  Shape: {J.shape}  (n_measurements x n_nodes)")
print(f"  Max sensitivity: {float(jnp.max(jnp.abs(J))):.6e}")
print(f"  Mostly negative: {bool(jnp.sum(J < 0) > J.size // 2)}")
print(f"  (negative = more absorption at node reduces detector signal)")

# --- Autodiff: d(detval)/d(mua) via jax.grad ---
print("\n=== Autodiff Gradient ===")


def total_signal(mua):
    r = forward_cw(mesh, mua, musp, srcpos, detpos, n_in, n_out)
    return jnp.sum(r.detval)


g = jax.grad(total_signal)(mua)
print(f"  d(sum_detval)/d(mua) = {float(g):.6e}")
print(f"  Compare with sum(J): {float(jnp.sum(J)):.6e}")
print(f"  (these are related but not identical — autodiff differentiates")
print(f"   through the full solve, Jacobian uses adjoint approximation)")
