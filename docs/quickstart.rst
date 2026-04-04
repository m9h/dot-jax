Quick Start
===========

This example runs a complete CW forward solve and computes the
autodiff gradient of the detector signal with respect to absorption.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from dot_jax.mesh import FEMMesh
   from dot_jax.forward import forward_cw

   # Build a tetrahedral mesh (20x20x20 mm box, 5 elements)
   node = jnp.array([
       [0, 0, 0], [20, 0, 0], [0, 20, 0], [20, 20, 0],
       [0, 0, 20], [20, 0, 20], [0, 20, 20], [20, 20, 20],
   ], dtype=jnp.float64)
   elem = jnp.array([
       [0,1,2,4], [1,2,3,7], [1,2,4,7], [1,4,5,7], [2,4,6,7],
   ], dtype=jnp.int32)
   mesh = FEMMesh.create(node, elem)

   # Forward solve
   srcpos = jnp.array([[5.0, 5.0, 5.0]])
   detpos = jnp.array([[15.0, 15.0, 15.0]])
   result = forward_cw(mesh, mua=0.01, musp=1.0, srcpos=srcpos, detpos=detpos)

   print(f"Detector value: {float(result.detval[0,0]):.4e}")

   # Autodiff gradient
   grad_mua = jax.grad(lambda mua: jnp.sum(
       forward_cw(mesh, mua, 1.0, srcpos, detpos).detval
   ))(0.01)
   print(f"d(signal)/d(mua) = {float(grad_mua):.4e}")

Pipeline overview
-----------------

The typical dot-jax workflow:

1. **Build mesh** — ``FEMMesh.create(node, elem)`` precomputes operators
2. **Set properties** — absorption (``mua``), scattering (``musp``), refractive indices
3. **Forward solve** — ``forward_cw()`` assembles, solves, and extracts detector values
4. **Differentiate** — ``jax.grad(forward_cw, ...)`` gives sensitivity to any parameter
5. **Reconstruct** — ``reconstruct_mua()`` inverts boundary data for spatial mua maps
