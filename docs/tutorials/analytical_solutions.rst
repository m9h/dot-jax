Analytical Solutions and Validation
=====================================

This tutorial covers the closed-form analytical solutions for the
photon diffusion equation in homogeneous media, implemented as
JAX-native functions in :mod:`dot_jax.analytical`. These solutions
serve two purposes: (1) standalone calculations for simple geometries,
and (2) ground truth for validating the FEM forward solver.

All functions are JIT-compatible and differentiable via ``jax.grad``,
enabling sensitivity analysis and gradient-based optimisation of
optical properties.

.. contents:: In this tutorial
   :local:
   :depth: 2


Prerequisites
-------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np
   import numpy.testing as npt

   jax.config.update("jax_enable_x64", True)

   from dot_jax.analytical import (
       getreff,
       getdistance,
       infinite_cw,
       semi_infinite_cw,
       spbesselj,
       spbessely,
       spbesselh,
       spbesseljprime,
       spbesselyprime,
       spbesselhprime,
       spharmonic,
   )


CW fluence in an infinite medium
----------------------------------

For an isotropic point source in a homogeneous infinite medium, the
steady-state (CW) fluence rate is the Green's function of the
diffusion equation:

.. math::

   \Phi(r) = \frac{1}{4\pi D}\,\frac{e^{-\mu_{\mathrm{eff}}\,r}}{r}

where:

- :math:`D = 1/[3(\mu_a + \mu_s')]` is the diffusion coefficient
- :math:`\mu_{\mathrm{eff}} = \sqrt{\mu_a / D}` is the effective
  attenuation coefficient
- :math:`r` is the source-detector distance

This solution decays exponentially with distance and inversely with
:math:`r` (geometric spreading).

.. code-block:: python

   mua = 0.01   # absorption (1/mm)
   musp = 1.0   # reduced scattering (1/mm)

   srcpos = jnp.array([[0.0, 0.0, 0.0]])
   detpos = jnp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 10.0, 0.0]])

   phi = infinite_cw(mua, musp, srcpos, detpos)
   assert jnp.all(phi > 0)  # fluence is always positive

**Radial decay:**

Fluence decreases monotonically with distance from the source:

.. code-block:: python

   det_radial = jnp.array([
       [5.0, 0.0, 0.0],
       [10.0, 0.0, 0.0],
       [20.0, 0.0, 0.0],
   ])
   phi = infinite_cw(mua, musp, srcpos, det_radial)
   assert phi[0] > phi[1] > phi[2]

**Spherical symmetry:**

Points at equal distance from the source have equal fluence,
regardless of direction:

.. code-block:: python

   det_sym = jnp.array([
       [10.0, 0.0, 0.0],
       [-10.0, 0.0, 0.0],
       [0.0, 10.0, 0.0],
   ])
   phi = infinite_cw(mua, musp, srcpos, det_sym)
   npt.assert_allclose(phi[0], phi[1], rtol=1e-10)
   npt.assert_allclose(phi[0], phi[2], rtol=1e-10)


CW fluence in a semi-infinite medium
--------------------------------------

For tissue with a planar boundary at :math:`z = 0` (e.g. the scalp
surface), the extrapolated boundary condition is implemented using the
method of image sources (Haskell et al. 1994; Farrell et al. 1992):

.. math::

   \Phi(r) = \frac{1}{4\pi D}\left[
   \frac{e^{-\mu_{\mathrm{eff}}\,r_1}}{r_1}
   - \frac{e^{-\mu_{\mathrm{eff}}\,r_2}}{r_2}
   \right]

where:

- :math:`r_1` is the distance to the real source (displaced :math:`z_0 = 1/(\mu_a + \mu_s')` into the medium)
- :math:`r_2` is the distance to the image source at :math:`z = -(z_0 + 2z_b)`
- :math:`z_b = 2D(1 + R_{\mathrm{eff}})/(1 - R_{\mathrm{eff}})` is the extrapolated boundary distance

The image source enforces the extrapolated boundary condition, and the
negative sign ensures :math:`\Phi \to 0` at :math:`z = -z_b`.

.. code-block:: python

   n_in, n_out = 1.37, 1.0  # tissue / air
   src_surf = jnp.array([[30.0, 30.0, 0.0]])
   det_surf = jnp.array([[30.0, 40.0, 0.0]])

   phi_inf = infinite_cw(mua, musp, src_surf, det_surf)
   phi_semi = semi_infinite_cw(mua, musp, n_in, n_out, src_surf, det_surf)

   # Semi-infinite fluence is always less than infinite (photons escape)
   assert phi_semi < phi_inf

**Multiple detectors:**

.. code-block:: python

   det_multi = jnp.array([[30.0, 40.0, 0.0], [40.0, 30.0, 0.0]])
   phi = semi_infinite_cw(mua, musp, n_in, n_out, src_surf, det_multi)
   assert phi.shape == (2,)


Effective reflection coefficient
----------------------------------

The effective reflection coefficient :math:`R_{\mathrm{eff}}` quantifies
internal reflection at the tissue-air boundary. It is computed by
numerical integration of the Fresnel reflectance over all angles
(Haskell 1994):

.. code-block:: python

   # Matched refractive index -> no internal reflection
   assert getreff(1.0, 1.0) == 0.0

   # Tissue-air interface (n=1.37)
   Reff = getreff(1.37, 1.0)
   assert 0 < Reff < 1  # partial internal reflection

   # Higher mismatch -> higher reflection
   Reff_low = getreff(1.1, 1.0)
   Reff_high = getreff(1.5, 1.0)
   assert Reff_high > Reff_low

.. note::

   ``getreff`` is JIT-compatible and uses ``jax.lax.cond`` for the
   matched-index special case. It can be differentiated with respect to
   ``n_in`` for refractive index sensitivity analysis.


Source-detector distance matrix
---------------------------------

The utility function :func:`~dot_jax.analytical.getdistance` computes
the Euclidean distance matrix between sources and detectors:

.. code-block:: python

   src = jnp.array([[0.0, 0.0, 0.0]])
   det = jnp.array([[3.0, 4.0, 0.0]])
   d = getdistance(src, det)
   npt.assert_allclose(d, 5.0, atol=1e-10)  # 3-4-5 triangle

   # Multiple detectors
   det_multi = jnp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
   d = getdistance(src, det_multi)
   assert d.shape == (2, 1)  # (n_det, n_src)
   npt.assert_allclose(d[:, 0], [10.0, 20.0], atol=1e-10)


Spherical Bessel functions
---------------------------

Spherical Bessel functions are the radial solutions to the
Helmholtz equation in spherical coordinates. They arise in
analytical DOT solutions for spherical geometries (e.g. sphere
phantoms, head models).

dot-jax provides JAX-native implementations using upward recurrence,
avoiding scipy calls so they work inside ``jax.jit`` and
``jax.grad``.

**Spherical Bessel function of the first kind** :math:`j_n(z)`:

.. math::

   j_0(z) = \frac{\sin z}{z}, \qquad
   j_1(z) = \frac{\sin z}{z^2} - \frac{\cos z}{z}

Higher orders use the recurrence:
:math:`j_{n+1}(z) = \frac{2n+1}{z}\,j_n(z) - j_{n-1}(z)`

.. code-block:: python

   z = 1.0
   j0 = spbesselj(0, z)
   npt.assert_allclose(j0, jnp.sin(z) / z, atol=1e-10)

   z = 2.0
   j1 = spbesselj(1, z)
   expected = jnp.sin(z) / z**2 - jnp.cos(z) / z
   npt.assert_allclose(j1, expected, atol=1e-10)

**Cross-validation with scipy:**

.. code-block:: python

   from scipy.special import spherical_jn

   for n in range(6):
       for z_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
           npt.assert_allclose(
               float(spbesselj(n, z_val)),
               float(spherical_jn(n, z_val)),
               rtol=1e-6,
           )

**Spherical Bessel function of the second kind** (Neumann function)
:math:`y_n(z)`:

.. math::

   y_0(z) = -\frac{\cos z}{z}

.. code-block:: python

   z = 1.0
   y0 = spbessely(0, z)
   npt.assert_allclose(y0, -jnp.cos(z) / z, atol=1e-10)


Spherical Hankel functions
---------------------------

Spherical Hankel functions represent outgoing and incoming spherical
waves. They are complex combinations of the first and second kind
Bessel functions:

.. math::

   h_n^{(1)}(z) = j_n(z) + i\,y_n(z), \qquad
   h_n^{(2)}(z) = j_n(z) - i\,y_n(z)

.. code-block:: python

   z = 1.5
   for n in range(4):
       h1 = spbesselh(n, 1, z)
       expected = spbesselj(n, z) + 1j * spbessely(n, z)
       npt.assert_allclose(h1, expected, atol=1e-10)

       h2 = spbesselh(n, 2, z)
       expected = spbesselj(n, z) - 1j * spbessely(n, z)
       npt.assert_allclose(h2, expected, atol=1e-10)


Derivatives of spherical Bessel functions
------------------------------------------

Derivatives are needed for boundary conditions in spherical geometry
problems. They use the recurrence:

.. math::

   j_n'(z) = j_{n-1}(z) - \frac{n+1}{z}\,j_n(z) \quad (n \geq 1),
   \qquad j_0'(z) = -j_1(z)

.. code-block:: python

   z = 1.0

   # j'_0(z) = -j_1(z)
   npt.assert_allclose(
       spbesseljprime(0, z), -spbesselj(1, z), atol=1e-10,
   )

   # Cross-validate y' with scipy
   from scipy.special import spherical_yn
   expected = spherical_yn(0, 2.0, derivative=True)
   npt.assert_allclose(float(spbesselyprime(0, 2.0)), float(expected), rtol=1e-12)

   # Hankel derivative: h'^(1)_n = j'_n + i*y'_n
   h1_prime = spbesselhprime(0, 1, z)
   expected = spbesseljprime(0, z) + 1j * spbesselyprime(0, z)
   npt.assert_allclose(h1_prime, expected, atol=1e-10)


Spherical harmonics
--------------------

Spherical harmonics :math:`Y_l^m(\theta, \phi)` describe the angular
dependence of solutions in spherical coordinates. They are used in
the series expansion of Green's functions for layered sphere models.

.. math::

   Y_l^m(\theta, \phi) = \sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}}
   \,P_l^m(\cos\theta)\,e^{im\phi}

.. code-block:: python

   # Y_0^0 is a constant: 1/sqrt(4*pi)
   Y00 = spharmonic(0, 0, 0.5, 0.0)
   npt.assert_allclose(jnp.real(Y00), 1.0 / jnp.sqrt(4 * jnp.pi), atol=1e-10)

   # Y_1^0(theta, 0) = sqrt(3/(4*pi)) * cos(theta)
   theta = jnp.pi / 4
   Y10 = spharmonic(1, 0, theta, 0.0)
   expected = jnp.sqrt(3.0 / (4.0 * jnp.pi)) * jnp.cos(theta)
   npt.assert_allclose(jnp.real(Y10), expected, atol=1e-10)


Autodiff through analytical models
------------------------------------

A key advantage of the JAX-native implementation is that all
analytical solutions are differentiable. This enables gradient-based
fitting of optical properties to measured data without finite
differences.

**Gradient of infinite-medium fluence w.r.t. absorption:**

.. code-block:: python

   def phi_sum_mua(mua):
       return jnp.sum(infinite_cw(
           mua, 1.0,
           jnp.array([[0.0, 0.0, 0.0]]),
           jnp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
       ))

   grad_mua = jax.grad(phi_sum_mua)(0.01)
   assert jnp.isfinite(grad_mua)
   assert grad_mua < 0  # more absorption -> less fluence

**Gradient w.r.t. scattering:**

.. code-block:: python

   def phi_sum_musp(musp):
       return jnp.sum(infinite_cw(
           0.01, musp,
           jnp.array([[0.0, 0.0, 0.0]]),
           jnp.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
       ))

   grad_musp = jax.grad(phi_sum_musp)(1.0)
   assert jnp.isfinite(grad_musp)

**Gradient of semi-infinite fluence:**

.. code-block:: python

   def phi_semi_mua(mua):
       return jnp.sum(semi_infinite_cw(
           mua, 1.0, 1.37, 1.0,
           jnp.array([[30.0, 30.0, 0.0]]),
           jnp.array([[30.0, 40.0, 0.0]]),
       ))

   grad_semi = jax.grad(phi_semi_mua)(0.01)
   assert jnp.isfinite(grad_semi)
   assert grad_semi < 0

**JIT compilation:**

.. code-block:: python

   f_jit = jax.jit(lambda mua, musp: infinite_cw(
       mua, musp,
       jnp.array([[0.0, 0.0, 0.0]]),
       jnp.array([[10.0, 0.0, 0.0]]),
   ))

   phi_eager = infinite_cw(0.01, 1.0,
       jnp.array([[0.0, 0.0, 0.0]]),
       jnp.array([[10.0, 0.0, 0.0]]),
   )
   phi_jitted = f_jit(0.01, 1.0)
   npt.assert_allclose(phi_jitted, phi_eager, rtol=1e-12)

**vmap over detectors:**

.. code-block:: python

   src = jnp.array([[0.0, 0.0, 0.0]])
   dets = jax.random.uniform(jax.random.PRNGKey(0), (100, 3), minval=1.0, maxval=50.0)

   def single_det(det):
       return infinite_cw(0.01, 1.0, src, det[None, :])[0]

   phi_batch = jax.vmap(single_det)(dets)
   assert phi_batch.shape == (100,) or phi_batch.shape == (100, 1)
   assert jnp.all(jnp.isfinite(phi_batch))

**vmap over Bessel function arguments:**

.. code-block:: python

   z_vals = jnp.linspace(0.1, 10.0, 50)
   j0_batch = jax.vmap(lambda zi: spbesselj(0, zi))(z_vals)
   assert j0_batch.shape == (50,)
   assert jnp.all(jnp.isfinite(j0_batch))


Comparing FEM to analytical solutions
---------------------------------------

The primary use of analytical solutions is validating the FEM forward
solver. Here is the workflow for a convergence study:

.. code-block:: python

   from dot_jax.mesh import FEMMesh
   from dot_jax.forward import forward_cw

   # Create a simple box mesh
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

   srcpos = jnp.array([[5.0, 5.0, 5.0]])
   detpos = jnp.array([[15.0, 15.0, 15.0]])

   # FEM solution
   result_fem = forward_cw(mesh, 0.01, 1.0, srcpos, detpos)

   # Analytical solution (infinite medium approximation)
   phi_analytical = infinite_cw(0.01, 1.0, srcpos, detpos)

   # The FEM solution on this coarse mesh will differ from the
   # infinite-medium analytical solution (boundary effects,
   # discretisation error), but both should be positive and finite
   assert jnp.all(result_fem.detval > 0)
   assert jnp.all(phi_analytical > 0)

.. note::

   For a rigorous convergence study, refine the mesh and observe that
   the FEM solution approaches the analytical solution as
   :math:`h \to 0`. The FEM solution also includes boundary effects
   that the infinite-medium solution does not, so for fair comparison,
   place sources and detectors far from the boundary or use the
   semi-infinite analytical solution.


API reference
--------------

- :func:`dot_jax.analytical.getreff`
- :func:`dot_jax.analytical.getdistance`
- :func:`dot_jax.analytical.infinite_cw`
- :func:`dot_jax.analytical.semi_infinite_cw`
- :func:`dot_jax.analytical.spbesselj`
- :func:`dot_jax.analytical.spbessely`
- :func:`dot_jax.analytical.spbesselh`
- :func:`dot_jax.analytical.spbesseljprime`
- :func:`dot_jax.analytical.spbesselyprime`
- :func:`dot_jax.analytical.spbesselhprime`
- :func:`dot_jax.analytical.spharmonic`
