The DOT Forward Problem: FEM-Based Light Transport
====================================================

This tutorial covers the complete FEM-based forward model in dot-jax:
creating a tetrahedral mesh, assigning optical properties, assembling
the stiffness/mass/boundary system matrix, and solving the photon
diffusion equation to compute fluence fields and detector
measurements.

All operations are implemented in pure JAX and are differentiable
with respect to the optical properties (:math:`\mu_a`,
:math:`\mu_s'`), enabling autodiff Jacobians for image
reconstruction.

.. contents:: In this tutorial
   :local:
   :depth: 2


Prerequisites
-------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np

   jax.config.update("jax_enable_x64", True)

   from dot_jax.mesh import (
       FEMMesh,
       compute_evol,
       compute_face_area,
       compute_deldotdel,
       compute_nvol,
       elem2node,
       extract_surface,
       reorient_elems,
       smooth_on_mesh,
   )
   from dot_jax.assembly import (
       assemble_stiffness,
       assemble_mass,
       assemble_boundary,
       assemble_system_cw,
   )
   from dot_jax.forward import (
       locate_sources,
       assemble_rhs,
       get_detector_values,
       forward_cw,
       forward_cw_sparse,
   )


The photon diffusion equation
------------------------------

In the continuous-wave (CW, steady-state) regime, photon transport
in highly scattering tissue is well approximated by the diffusion
equation:

.. math::

   -\nabla \cdot \bigl[D(\mathbf{r})\,\nabla\Phi(\mathbf{r})\bigr]
   + \mu_a(\mathbf{r})\,\Phi(\mathbf{r}) = S(\mathbf{r})

where:

- :math:`\Phi(\mathbf{r})` is the photon fluence rate (W/mm\ :sup:`2`)
- :math:`D = 1/[3(\mu_a + \mu_s')]` is the diffusion coefficient
- :math:`\mu_a` is the absorption coefficient (1/mm)
- :math:`\mu_s'` is the reduced scattering coefficient (1/mm)
- :math:`S(\mathbf{r})` is the source term

At tissue-air boundaries, the Robin (type III) boundary condition
accounts for the refractive index mismatch:

.. math::

   \Phi + 2A\,D\,\frac{\partial\Phi}{\partial\hat{n}} = 0

where :math:`A = (1 + R_{\mathrm{eff}})/(1 - R_{\mathrm{eff}})`
and :math:`R_{\mathrm{eff}}` is the effective Fresnel reflection
coefficient (Haskell et al. 1994).


FEM weak formulation
---------------------

Discretising with linear tetrahedral elements and applying the Galerkin
method, the weak form becomes the linear system:

.. math::

   \mathbf{A}\,\boldsymbol{\Phi} = \mathbf{b}

where the system matrix decomposes as:

.. math::

   \mathbf{A} = \mathbf{K} + \mathbf{M} + \mathbf{C}

with:

- **Stiffness matrix** :math:`K_{ij} = \sum_e D_e \int_{\Omega_e} \nabla\phi_i \cdot \nabla\phi_j \,\mathrm{d}V`
  (diffusion)
- **Mass matrix** :math:`M_{ij} = \sum_e \mu_{a,e} \int_{\Omega_e} \phi_i\,\phi_j\,\mathrm{d}V`
  (absorption)
- **Boundary matrix** :math:`C_{ij} = \frac{1-R_{\mathrm{eff}}}{12(1+R_{\mathrm{eff}})} \sum_f \int_{\Gamma_f} \phi_i\,\phi_j\,\mathrm{d}A`
  (Robin BC)

For linear tetrahedra, the integrals have closed forms using element
volumes :math:`V_e` and the precomputed gradient dot products
:math:`\nabla\phi_i \cdot \nabla\phi_j`.


Step 1: Mesh creation
----------------------

A ``FEMMesh`` is an Equinox Module holding the mesh geometry and
precomputed operators. It is constructed from node coordinates and
element connectivity arrays (0-based indexing):

.. code-block:: python

   # 8-node cube, 5 tetrahedra (a standard minimal 3D box mesh)
   node = jnp.array([
       [0.0, 0.0, 0.0],
       [10.0, 0.0, 0.0],
       [0.0, 10.0, 0.0],
       [10.0, 10.0, 0.0],
       [0.0, 0.0, 10.0],
       [10.0, 0.0, 10.0],
       [0.0, 10.0, 10.0],
       [10.0, 10.0, 10.0],
   ], dtype=jnp.float64)

   elem = jnp.array([
       [0, 1, 2, 4],
       [1, 2, 3, 7],
       [1, 2, 4, 7],
       [1, 4, 5, 7],
       [2, 4, 6, 7],
   ], dtype=jnp.int32)

   mesh = FEMMesh.create(node, elem)

   assert mesh.nn == 8    # number of nodes
   assert mesh.ne == 5    # number of elements
   assert mesh.nf == 12   # number of boundary faces (auto-extracted)

``FEMMesh.create`` automatically:

1. Reorients elements to ensure positive volumes
2. Extracts the boundary surface (if ``face`` is not provided)
3. Computes element volumes, face areas, nodal volumes, and
   gradient dot products

**Element volumes:**

.. code-block:: python

   # Unit tetrahedron: V = 1/6
   node_unit = jnp.array([
       [0.0, 0.0, 0.0],
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
       [0.0, 0.0, 1.0],
   ])
   elem_unit = jnp.array([[0, 1, 2, 3]], dtype=jnp.int32)
   evol = compute_evol(node_unit, elem_unit)
   assert jnp.allclose(evol[0], 1.0 / 6.0, rtol=1e-12)

   # 10x10x10 box: total volume = 1000 mm^3
   assert jnp.allclose(jnp.sum(mesh.evol), 1000.0, rtol=1e-10)

**Volume scales with node coordinates:**

.. code-block:: python

   evol_1x = compute_evol(node_unit, elem_unit)
   evol_2x = compute_evol(2.0 * node_unit, elem_unit)
   assert jnp.allclose(evol_2x, 8.0 * evol_1x, rtol=1e-12)

**Face areas:**

.. code-block:: python

   # Right triangle with legs of length 1: area = 0.5
   face_node = jnp.array([
       [0.0, 0.0, 0.0],
       [1.0, 0.0, 0.0],
       [0.0, 1.0, 0.0],
   ])
   face_conn = jnp.array([[0, 1, 2]], dtype=jnp.int32)
   area = compute_face_area(face_node, face_conn)
   assert jnp.allclose(area[0], 0.5, rtol=1e-12)

**Nodal volumes (conservation):**

Nodal volumes are computed by distributing 1/4 of each element's
volume to its four vertices. The sum of nodal volumes equals the
total mesh volume:

.. code-block:: python

   assert jnp.allclose(jnp.sum(mesh.nvol), jnp.sum(mesh.evol), rtol=1e-10)

**FEMMesh as a JAX pytree:**

Because ``FEMMesh`` is an Equinox Module, it can be passed through
JAX transformations (``jit``, ``vmap``, ``grad``):

.. code-block:: python

   leaves = jax.tree.leaves(mesh)
   assert len(leaves) == 7  # node, elem, face, evol, area, nvol, deldotdel


Step 2: Gradient dot products (deldotdel)
------------------------------------------

The core FEM assembly data is the matrix of gradient dot products
:math:`\nabla\phi_i \cdot \nabla\phi_j` for each element. For a
linear tetrahedron with 4 basis functions, there are 10 unique
products (upper triangle of a 4x4 symmetric matrix).

.. code-block:: python

   evol = compute_evol(node, elem)
   dd, delphi = compute_deldotdel(node, elem, evol)

   assert dd.shape == (5, 10)       # (ne, 10) upper-triangle products
   assert delphi.shape == (5, 3, 4) # (ne, 3, 4) gradient components

   # Diagonal entries (self dot products) are always positive
   diag_idx = [0, 4, 7, 9]  # (0,0), (1,1), (2,2), (3,3)
   for idx in diag_idx:
       assert jnp.all(dd[:, idx] > 0)

.. note::

   The deldotdel computation ports the redbirdpy algorithm exactly
   and is cross-validated against it in the test suite.

**Differentiability:** The gradient products are differentiable with
respect to node coordinates, enabling mesh sensitivity analysis:

.. code-block:: python

   def scalar_dd(node):
       evol = compute_evol(node, elem)
       dd, _ = compute_deldotdel(node, elem, evol)
       return jnp.sum(dd)

   g = jax.grad(scalar_dd)(node)
   assert jnp.all(jnp.isfinite(g))
   assert g.shape == node.shape


Step 3: Stiffness matrix assembly
-----------------------------------

The stiffness matrix :math:`\mathbf{K}` encodes diffusion. It is
symmetric positive semi-definite and scales linearly with the
diffusion coefficient :math:`D`.

.. code-block:: python

   mua, musp = 0.01, 1.0
   D = 1.0 / (3.0 * (mua + musp))

   K = assemble_stiffness(mesh, D)

   assert K.shape == (mesh.nn, mesh.nn)

   # Symmetric
   assert jnp.allclose(K, K.T, atol=1e-15)

   # Positive semi-definite
   eigvals = jnp.linalg.eigvalsh(K)
   assert jnp.all(eigvals >= -1e-12)

   # Linear scaling with D
   K2 = assemble_stiffness(mesh, 2.0 * D)
   assert jnp.allclose(K2, 2.0 * K, rtol=1e-12)

:func:`~dot_jax.assembly.assemble_stiffness` accepts either a scalar
:math:`D` (homogeneous medium) or a per-element array ``(ne,)`` for
heterogeneous media:

.. code-block:: python

   D_array = jnp.full(mesh.ne, D)
   K_array = assemble_stiffness(mesh, D_array)
   assert jnp.allclose(K, K_array, atol=1e-15)


Step 4: Mass matrix assembly
------------------------------

The consistent mass matrix :math:`\mathbf{M}` represents absorption.
For linear tetrahedra, the consistent mass coefficients are 1/10
(diagonal) and 1/20 (off-diagonal).

.. code-block:: python

   M = assemble_mass(mesh, mua)

   assert M.shape == (mesh.nn, mesh.nn)
   assert jnp.allclose(M, M.T, atol=1e-15)

   # Positive semi-definite
   eigvals = jnp.linalg.eigvalsh(M)
   assert jnp.all(eigvals >= -1e-12)

   # Linear scaling with mua
   M2 = assemble_mass(mesh, 2.0 * mua)
   assert jnp.allclose(M2, 2.0 * M, rtol=1e-12)

**Row sum property:** When :math:`\mu_a = 1`, the row sums of the
consistent mass matrix equal the nodal volumes:

.. code-block:: python

   M_unit = assemble_mass(mesh, 1.0)
   row_sums = jnp.sum(M_unit, axis=1)
   assert jnp.allclose(row_sums, mesh.nvol, rtol=1e-12)


Step 5: Robin boundary condition
---------------------------------

The boundary matrix :math:`\mathbf{C}` enforces the Robin boundary
condition at the tissue-air interface. It depends on the refractive
index mismatch through the effective reflection coefficient
:math:`R_{\mathrm{eff}}`.

.. code-block:: python

   n_in, n_out = 1.37, 1.0  # tissue / air
   C = assemble_boundary(mesh, n_in, n_out)

   assert C.shape == (mesh.nn, mesh.nn)
   assert jnp.allclose(C, C.T, atol=1e-15)

   # Positive semi-definite
   eigvals = jnp.linalg.eigvalsh(C)
   assert jnp.all(eigvals >= -1e-12)

**Effect of refractive index mismatch:**

A larger index mismatch produces a smaller boundary contribution
(photons are more strongly reflected back into the tissue):

.. code-block:: python

   C_low = assemble_boundary(mesh, 1.1, 1.0)
   C_high = assemble_boundary(mesh, 1.5, 1.0)
   assert jnp.sum(jnp.abs(C_high)) < jnp.sum(jnp.abs(C_low))


Step 6: Full system matrix
----------------------------

The complete CW system matrix combines all three terms:

.. math::

   \mathbf{A} = \mathbf{K} + \mathbf{M} + \mathbf{C}

.. code-block:: python

   A = assemble_system_cw(mesh, mua, musp, n_in, n_out)

   assert A.shape == (mesh.nn, mesh.nn)
   assert jnp.allclose(A, A.T, atol=1e-14)

   # Positive definite (with both absorption and boundary condition)
   eigvals = jnp.linalg.eigvalsh(A)
   assert jnp.all(eigvals > 0)

   # Verify A = K + M + C
   K = assemble_stiffness(mesh, D)
   M = assemble_mass(mesh, mua)
   C = assemble_boundary(mesh, n_in, n_out)
   assert jnp.allclose(A, K + M + C, atol=1e-14)

**Differentiability:**

.. code-block:: python

   def a_sum(mua, musp):
       return jnp.sum(assemble_system_cw(mesh, mua, musp, 1.37, 1.0))

   g_mua = jax.grad(a_sum, argnums=0)(0.01, 1.0)
   g_musp = jax.grad(a_sum, argnums=1)(0.01, 1.0)
   assert jnp.isfinite(g_mua)
   assert jnp.isfinite(g_musp)


Step 7: Source and detector placement
--------------------------------------

Sources and detectors are placed at arbitrary positions using
barycentric interpolation into the mesh elements.

**Locating points in the mesh:**

.. code-block:: python

   # 20x20x20 mm box for realistic optical properties
   node_box = jnp.array([
       [0.0, 0.0, 0.0], [20.0, 0.0, 0.0],
       [0.0, 20.0, 0.0], [20.0, 20.0, 0.0],
       [0.0, 0.0, 20.0], [20.0, 0.0, 20.0],
       [0.0, 20.0, 20.0], [20.0, 20.0, 20.0],
   ], dtype=jnp.float64)
   elem_box = jnp.array([
       [0, 1, 2, 4], [1, 2, 3, 7], [1, 2, 4, 7],
       [1, 4, 5, 7], [2, 4, 6, 7],
   ], dtype=jnp.int32)
   box_mesh = FEMMesh.create(node_box, elem_box)

   srcpos = jnp.array([[5.0, 5.0, 5.0]])
   elem_idx, bary = locate_sources(box_mesh, srcpos)

   # Barycentric coordinates sum to 1 and are non-negative
   assert jnp.allclose(jnp.sum(bary, axis=1), 1.0, atol=1e-10)
   assert jnp.all(bary >= -1e-10)

**Assembling the RHS vector:**

Each source produces a column in the RHS matrix, with non-zero
entries only at the nodes of the containing element:

.. code-block:: python

   rhs = assemble_rhs(box_mesh, srcpos)
   assert rhs.shape == (box_mesh.nn, 1)

   # Column sums to 1 (partition of unity)
   assert jnp.allclose(jnp.sum(rhs[:, 0]), 1.0, atol=1e-10)

   # Non-negative entries
   assert jnp.all(rhs >= -1e-10)


Step 8: Solving the forward problem
-------------------------------------

The complete forward solve pipeline is wrapped in
:func:`~dot_jax.forward.forward_cw`:

.. code-block:: python

   srcpos = jnp.array([[5.0, 5.0, 5.0]])
   detpos = jnp.array([[15.0, 15.0, 15.0]])

   result = forward_cw(box_mesh, mua=0.01, musp=1.0, srcpos=srcpos, detpos=detpos)

   from dot_jax import ForwardResult
   assert isinstance(result, ForwardResult)
   assert result.phi.shape == (box_mesh.nn, 1)   # fluence at all nodes
   assert result.detval.shape == (1, 1)           # (n_det, n_src)

   # Fluence is finite and detector values are positive
   assert jnp.all(jnp.isfinite(result.phi))
   assert jnp.all(result.detval > 0)

**Multiple sources and detectors:**

.. code-block:: python

   srcpos = jnp.array([[3.0, 3.0, 3.0], [17.0, 17.0, 17.0]])
   detpos = jnp.array([[10.0, 10.0, 10.0], [15.0, 5.0, 5.0]])

   result = forward_cw(box_mesh, 0.01, 1.0, srcpos, detpos)
   assert result.detval.shape == (2, 2)   # (n_det, n_src)
   assert result.phi.shape == (box_mesh.nn, 2)  # one fluence column per source

**Physical consistency: more absorption means less signal:**

.. code-block:: python

   r_low = forward_cw(box_mesh, 0.01, 1.0, srcpos[:1], detpos[:1])
   r_high = forward_cw(box_mesh, 0.05, 1.0, srcpos[:1], detpos[:1])
   assert r_high.detval[0, 0] < r_low.detval[0, 0]


Step 9: Detector value extraction (adjoint method)
----------------------------------------------------

Detector values are computed via the adjoint (reciprocity) method:

.. math::

   d_{ij} = \mathbf{b}_{\mathrm{det},j}^T\,\boldsymbol{\Phi}_{\mathrm{src},i}

This is equivalent to solving a separate forward problem for each
detector as a virtual source and evaluating the overlap integral.
In matrix form:

.. code-block:: python

   phi = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
   rhs_det = jnp.array([[0.1, 0.0], [0.0, 0.2], [0.3, 0.0]])

   detval = get_detector_values(phi, rhs_det)
   expected = rhs_det.T @ phi
   assert jnp.allclose(detval, expected, atol=1e-15)


Step 10: Sparse CG solver for large meshes
--------------------------------------------

For meshes with more than ~10K nodes, the dense LU solver in
``forward_cw`` becomes memory-limited. The sparse CG solver uses
BCSR sparse format and conjugate gradients (the system is SPD):

.. code-block:: python

   result_sparse = forward_cw_sparse(box_mesh, 0.01, 1.0, srcpos[:1], detpos[:1])

   # Should agree with the dense solver
   result_dense = forward_cw(box_mesh, 0.01, 1.0, srcpos[:1], detpos[:1])
   assert jnp.allclose(result_sparse.detval, result_dense.detval, rtol=1e-5)
   assert jnp.allclose(result_sparse.phi, result_dense.phi, rtol=1e-5)

.. tip::

   Use ``forward_cw_sparse`` for meshes with more than ~10K nodes
   (typical brain DOT meshes have 50K-200K nodes). Memory scales as
   :math:`O(\mathrm{nnz})` instead of :math:`O(n^2)`.


Autodiff through the forward model
------------------------------------

The entire forward pipeline is differentiable with respect to
:math:`\mu_a` and :math:`\mu_s'`. This is the key advantage of
dot-jax: Jacobians for reconstruction are computed exactly via
``jax.grad`` rather than finite differences.

.. code-block:: python

   srcpos = jnp.array([[5.0, 5.0, 5.0]])
   detpos = jnp.array([[15.0, 15.0, 15.0]])

   def detval_sum(mua):
       r = forward_cw(box_mesh, mua, 1.0, srcpos, detpos)
       return jnp.sum(r.detval)

   # Gradient w.r.t. absorption
   g_mua = jax.grad(detval_sum)(0.01)
   assert jnp.isfinite(g_mua)
   assert g_mua < 0  # more absorption -> less signal

   # Gradient w.r.t. scattering
   def detval_sum_musp(musp):
       r = forward_cw(box_mesh, 0.01, musp, srcpos, detpos)
       return jnp.sum(r.detval)

   g_musp = jax.grad(detval_sum_musp)(1.0)
   assert jnp.isfinite(g_musp)

Both the dense (LU) and sparse (CG) solvers support autodiff, and
their gradients agree:

.. code-block:: python

   def f_dense(mua):
       return jnp.sum(forward_cw(box_mesh, mua, 1.0, srcpos, detpos).detval)

   def f_sparse(mua):
       return jnp.sum(forward_cw_sparse(box_mesh, mua, 1.0, srcpos, detpos).detval)

   g_dense = jax.grad(f_dense)(0.01)
   g_sparse = jax.grad(f_sparse)(0.01)
   assert jnp.allclose(g_sparse, g_dense, rtol=1e-3)


Utility: elem2node and spatial smoothing
------------------------------------------

**Scatter element values to nodes:**

.. code-block:: python

   elemval = jnp.ones(mesh.ne) * 4.0
   nodeval = elem2node(mesh.elem, elemval, mesh.nn)
   assert nodeval.shape == (mesh.nn,)
   assert jnp.all(nodeval > 0)

**Gaussian smoothing on the mesh:**

.. code-block:: python

   values = jax.random.normal(jax.random.PRNGKey(0), (mesh.nn,))
   smoothed = smooth_on_mesh(mesh, values, fwhm=5.0)
   assert smoothed.shape == (mesh.nn,)

   # Smoothing reduces spatial variance
   assert jnp.std(smoothed) < jnp.std(values)

   # Constant field is unchanged
   smoothed_const = smooth_on_mesh(mesh, jnp.ones(mesh.nn) * 7.0, fwhm=5.0)
   assert jnp.allclose(smoothed_const, 7.0, atol=1e-10)


API reference
--------------

Mesh:

- :class:`dot_jax.mesh.FEMMesh`
- :func:`dot_jax.mesh.compute_evol`
- :func:`dot_jax.mesh.compute_face_area`
- :func:`dot_jax.mesh.compute_deldotdel`
- :func:`dot_jax.mesh.compute_nvol`
- :func:`dot_jax.mesh.elem2node`
- :func:`dot_jax.mesh.extract_surface`
- :func:`dot_jax.mesh.reorient_elems`
- :func:`dot_jax.mesh.smooth_on_mesh`

Assembly:

- :func:`dot_jax.assembly.assemble_stiffness`
- :func:`dot_jax.assembly.assemble_mass`
- :func:`dot_jax.assembly.assemble_boundary`
- :func:`dot_jax.assembly.assemble_system_cw`

Forward:

- :func:`dot_jax.forward.forward_cw`
- :func:`dot_jax.forward.forward_cw_sparse`
- :func:`dot_jax.forward.locate_sources`
- :func:`dot_jax.forward.assemble_rhs`
- :func:`dot_jax.forward.get_detector_values`
