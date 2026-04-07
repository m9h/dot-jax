DOT Image Reconstruction: Gauss-Newton Inversion
===================================================

This tutorial covers the inverse problem in diffuse optical
tomography: recovering the spatial distribution of absorption
coefficients from boundary measurements. dot-jax implements
linearised Gauss-Newton inversion with Tikhonov regularisation,
leveraging JAX autodiff for exact Jacobian computation.

.. contents:: In this tutorial
   :local:
   :depth: 2


Prerequisites
-------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy.testing as npt

   jax.config.update("jax_enable_x64", True)

   from dot_jax.mesh import FEMMesh
   from dot_jax.forward import forward_cw
   from dot_jax.recon import (
       reconstruct_mua,
       solve_dual,
       compute_lcurve,
       select_lambda_lcurve,
       select_lambda_gcv,
   )
   from dot_jax.spectral import compute_jacobian_mua


The DOT inverse problem
------------------------

The forward problem maps optical properties to boundary measurements:

.. math::

   \mathbf{d} = \mathcal{F}(\boldsymbol{\mu}_a, \mu_s')

The inverse problem seeks to recover the nodal absorption coefficient
:math:`\boldsymbol{\mu}_a` from measured data :math:`\mathbf{d}_{\mathrm{meas}}`.
This is an ill-posed problem -- there are far more unknowns (mesh nodes)
than measurements (source-detector pairs), so regularisation is essential.

**Linearisation (Born approximation):**

Near a background operating point :math:`\mu_a^{(0)}`, the forward
model is linearised:

.. math::

   \Delta\mathbf{d} \approx \mathbf{J}\,\Delta\boldsymbol{\mu}_a

where :math:`\mathbf{J}` is the Jacobian (sensitivity) matrix:

.. math::

   J_{ij} = \frac{\partial d_i}{\partial \mu_{a,j}}

In dot-jax, the Jacobian is computed via the adjoint method
(Arridge 1999):

.. math::

   J[d,s,n] = -\Phi_{\mathrm{src}}(n,s)\,\Phi_{\mathrm{det}}(n,d)\,V_n

where :math:`\Phi_{\mathrm{src}}` and :math:`\Phi_{\mathrm{det}}`
are the forward and adjoint Green's functions, and :math:`V_n` is
the nodal volume.


Setting up a reconstruction problem
-------------------------------------

First, create a mesh and generate synthetic data with a known
absorption perturbation:

.. code-block:: python

   # 20x20x20 mm box mesh
   node = jnp.array([
       [0.0, 0.0, 0.0], [20.0, 0.0, 0.0],
       [0.0, 20.0, 0.0], [20.0, 20.0, 0.0],
       [0.0, 0.0, 20.0], [20.0, 0.0, 20.0],
       [0.0, 20.0, 20.0], [20.0, 20.0, 20.0],
   ], dtype=jnp.float64)
   elem = jnp.array([
       [0, 1, 2, 4], [1, 2, 3, 7], [1, 2, 4, 7],
       [1, 4, 5, 7], [2, 4, 6, 7],
   ], dtype=jnp.int32)
   mesh = FEMMesh.create(node, elem)

   # Source-detector geometry
   srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
   detpos = jnp.array([[10.0, 10.0, 10.0], [17.0, 3.0, 3.0]])

   # Optical properties
   mua_bg = 0.01      # background absorption (1/mm)
   mua_true = 0.015   # true (perturbed) absorption
   musp = 1.0         # reduced scattering (1/mm), fixed during recon

   # Generate "measured" data at the true absorption
   result_true = forward_cw(mesh, mua_true, musp, srcpos, detpos)
   data = result_true.detval  # (n_det, n_src) = (2, 2)


Gauss-Newton reconstruction
-----------------------------

:func:`~dot_jax.recon.reconstruct_mua` implements iterative
Gauss-Newton inversion with Levenberg-Marquardt (Tikhonov)
regularisation. At each step:

.. math::

   \Delta\boldsymbol{\mu}_a =
   \bigl(\mathbf{J}^T\mathbf{J}
   + \lambda\,\mathrm{diag}(\mathbf{J}^T\mathbf{J})\bigr)^{-1}
   \mathbf{J}^T\,(\mathbf{d}_{\mathrm{meas}} - \mathbf{d}_{\mathrm{pred}})

The regularisation parameter :math:`\lambda` controls the trade-off
between data fidelity and solution smoothness.

**Single Gauss-Newton step:**

.. code-block:: python

   from dot_jax import ReconResult

   result = reconstruct_mua(
       mesh, data, srcpos, detpos,
       mua0=mua_bg,
       musp=musp,
       max_steps=1,
       reg_param=1e-4,
   )

   assert isinstance(result, ReconResult)
   assert result.mua.shape == (mesh.nn,)          # nodal absorption map
   assert result.residuals.shape == (2,)           # initial + final residual
   assert result.residuals[-1] < result.residuals[0]  # residual decreased

**Multi-step iteration:**

.. code-block:: python

   result_3 = reconstruct_mua(
       mesh, data, srcpos, detpos,
       mua0=mua_bg, musp=musp,
       max_steps=3,
       reg_param=1e-4,
   )

   assert result_3.residuals.shape == (4,)   # 3 steps + final
   assert jnp.all(jnp.isfinite(result_3.mua))
   assert jnp.all(jnp.isfinite(result_3.musp))

**No-change test:** When the data matches the background, the
reconstruction should produce no significant update:

.. code-block:: python

   data_bg = forward_cw(mesh, mua_bg, musp, srcpos[:1], detpos[:1]).detval
   result_bg = reconstruct_mua(
       mesh, data_bg, srcpos[:1], detpos[:1],
       mua0=mua_bg, musp=musp,
   )
   npt.assert_allclose(result_bg.mua, mua_bg, atol=1e-8)


The Jacobian (sensitivity matrix)
----------------------------------

The Jacobian :math:`\mathbf{J}` maps nodal absorption perturbations
to measurement changes. It is computed via the adjoint method in
:func:`~dot_jax.spectral.compute_jacobian_mua`:

.. code-block:: python

   J = compute_jacobian_mua(mesh, mua_bg, musp, srcpos, detpos)

   n_meas = srcpos.shape[0] * detpos.shape[0]  # 2 * 2 = 4
   assert J.shape == (n_meas, mesh.nn)  # (4, 8)

The Jacobian rows are ordered detector-major: row
:math:`d \cdot n_{\mathrm{src}} + s` corresponds to detector
:math:`d` and source :math:`s`.

.. note::

   The adjoint Jacobian is exact (no finite differences) and fully
   differentiable via JAX. This means you can compute second-order
   sensitivities (Hessians) for uncertainty quantification.


Dual-formulation solve for underdetermined systems
----------------------------------------------------

In DOT, the number of unknowns (mesh nodes, typically 10K-200K)
vastly exceeds the number of measurements (source-detector pairs,
typically 100-1000). The dual formulation operates in measurement
space rather than parameter space, which is more efficient:

.. math::

   \mathbf{x} = \mathbf{J}^T
   \bigl(\mathbf{J}\mathbf{J}^T
   + \lambda\sqrt{\|\mathbf{J}\mathbf{J}^T\|}\,\mathbf{I}\bigr)^{-1}
   \mathbf{d}

This solves an :math:`n_{\mathrm{meas}} \times n_{\mathrm{meas}}`
system instead of an :math:`n_{\mathrm{nodes}} \times n_{\mathrm{nodes}}`
one.

.. code-block:: python

   # Underdetermined system: 5 measurements, 100 unknowns
   J_test = jax.random.normal(jax.random.PRNGKey(0), (5, 100))
   d_test = jax.random.normal(jax.random.PRNGKey(1), (5,))

   x = solve_dual(J_test, d_test, reg=0.1)
   assert x.shape == (100,)
   assert jnp.all(jnp.isfinite(x))

**Signal recovery test:**

.. code-block:: python

   key = jax.random.PRNGKey(42)
   J_rec = jax.random.normal(key, (20, 8))
   x_true = jnp.ones(8) * 0.01
   d_rec = J_rec @ x_true

   x_recovered = solve_dual(J_rec, d_rec, reg=1e-6)
   npt.assert_allclose(x_recovered, x_true, atol=1e-3)


Regularisation parameter selection
------------------------------------

Choosing the right regularisation parameter :math:`\lambda` is
critical. Too small leads to noise amplification; too large
over-smooths the image. dot-jax provides two automatic methods.

L-curve method
^^^^^^^^^^^^^^

The L-curve plots the residual norm :math:`\|\mathbf{Jx} - \mathbf{d}\|`
against the solution norm :math:`\|\mathbf{x}\|` on a log-log scale.
The optimal :math:`\lambda` is at the point of maximum curvature
(the "corner").

.. code-block:: python

   key = jax.random.PRNGKey(42)
   J_lc = jax.random.normal(key, (20, 8))
   x_true_lc = jnp.ones(8) * 0.01
   d_lc = J_lc @ x_true_lc
   lambdas = jnp.logspace(-6, 2, 30)

   # Compute the L-curve
   res_norms, sol_norms = compute_lcurve(J_lc, d_lc, lambdas)
   assert res_norms.shape == (30,)
   assert sol_norms.shape == (30,)

   # Less regularization -> smaller residual, larger solution
   assert res_norms[-1] > res_norms[0]
   assert sol_norms[0] > sol_norms[-1]

   # Select optimal lambda at the L-curve corner
   lam_opt = select_lambda_lcurve(J_lc, d_lc, lambdas)
   assert jnp.isfinite(lam_opt)
   assert lam_opt > 0

**Automatic lambda range:** If no candidate range is provided,
``select_lambda_lcurve`` generates one from the singular values
of :math:`\mathbf{J}`:

.. code-block:: python

   lam_auto = select_lambda_lcurve(J_lc, d_lc)
   assert jnp.isfinite(lam_auto)
   assert lam_auto > 0

Generalised Cross-Validation (GCV)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GCV minimises the leave-one-out prediction error without requiring
explicit knowledge of the noise level:

.. math::

   \mathrm{GCV}(\lambda) =
   \frac{\|\mathbf{Jx}_\lambda - \mathbf{d}\|^2 / m}
   {\bigl[\mathrm{tr}(\mathbf{I} - \mathbf{H}_\lambda)/m\bigr]^2}

where :math:`\mathbf{H}_\lambda = \mathbf{J}(\mathbf{J}^T\mathbf{J}
+ \lambda\mathbf{I})^{-1}\mathbf{J}^T` is the hat (influence) matrix
and :math:`m` is the number of measurements.

.. code-block:: python

   lam_gcv = select_lambda_gcv(J_lc, d_lc, lambdas)
   assert jnp.isfinite(lam_gcv)
   assert lam_gcv > 0

   # Also works with automatic lambda range
   lam_gcv_auto = select_lambda_gcv(J_lc, d_lc)
   assert jnp.isfinite(lam_gcv_auto)

.. tip::

   GCV tends to perform better when the noise statistics are unknown.
   The L-curve method is more robust when the system is severely
   ill-conditioned.


Differentiable reconstruction through JAX
-------------------------------------------

Because the entire reconstruction pipeline -- forward solve, Jacobian
computation, and regularised inversion -- is built on JAX, you can
differentiate through the full pipeline. This enables:

- **End-to-end optimisation** of source/detector placement
- **Learned regularisation** via differentiable programming
- **Uncertainty quantification** via Fisher information / Hessian

The forward model is differentiable with respect to both
:math:`\mu_a` and :math:`\mu_s'`:

.. code-block:: python

   srcpos_single = jnp.array([[5.0, 5.0, 5.0]])
   detpos_single = jnp.array([[15.0, 15.0, 15.0]])

   def detval_sum(mua):
       r = forward_cw(mesh, mua, 1.0, srcpos_single, detpos_single)
       return jnp.sum(r.detval)

   # First-order gradient
   g = jax.grad(detval_sum)(0.01)
   assert jnp.isfinite(g)
   assert g < 0  # more absorption reduces signal

   # The sparse CG solver also supports autodiff
   from dot_jax.forward import forward_cw_sparse

   def f_sparse(mua):
       return jnp.sum(forward_cw_sparse(
           mesh, mua, 1.0, srcpos_single, detpos_single
       ).detval)

   g_sparse = jax.grad(f_sparse)(0.01)
   assert jnp.isfinite(g_sparse)
   assert g_sparse < 0


Complete reconstruction workflow
---------------------------------

Here is a self-contained workflow for DOT image reconstruction:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from dot_jax.mesh import FEMMesh
   from dot_jax.forward import forward_cw
   from dot_jax.recon import reconstruct_mua, select_lambda_gcv
   from dot_jax.spectral import compute_jacobian_mua

   # 1. Create mesh
   node = jnp.array([
       [0.0, 0.0, 0.0], [20.0, 0.0, 0.0],
       [0.0, 20.0, 0.0], [20.0, 20.0, 0.0],
       [0.0, 0.0, 20.0], [20.0, 0.0, 20.0],
       [0.0, 20.0, 20.0], [20.0, 20.0, 20.0],
   ], dtype=jnp.float64)
   elem = jnp.array([
       [0, 1, 2, 4], [1, 2, 3, 7], [1, 2, 4, 7],
       [1, 4, 5, 7], [2, 4, 6, 7],
   ], dtype=jnp.int32)
   mesh = FEMMesh.create(node, elem)

   # 2. Source-detector geometry
   srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
   detpos = jnp.array([[10.0, 10.0, 10.0], [17.0, 3.0, 3.0]])

   # 3. Generate synthetic data (simulate an absorption increase)
   mua_bg = 0.01
   mua_true = 0.015
   musp = 1.0
   data_meas = forward_cw(mesh, mua_true, musp, srcpos, detpos).detval

   # 4. Reconstruct
   result = reconstruct_mua(
       mesh, data_meas, srcpos, detpos,
       mua0=mua_bg, musp=musp,
       max_steps=3, reg_param=1e-4,
   )

   print(f"Reconstructed mua (mean): {float(jnp.mean(result.mua)):.6f}")
   print(f"True mua:                 {mua_true}")
   print(f"Final residual:           {float(result.residuals[-1]):.2e}")
   print(f"Residual reduction:       {float(result.residuals[0] / result.residuals[-1]):.1f}x")


API reference
--------------

- :func:`dot_jax.recon.reconstruct_mua`
- :func:`dot_jax.recon.solve_dual`
- :func:`dot_jax.recon.compute_lcurve`
- :func:`dot_jax.recon.select_lambda_lcurve`
- :func:`dot_jax.recon.select_lambda_gcv`
- :func:`dot_jax.spectral.compute_jacobian_mua`
