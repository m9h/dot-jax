Research Background
===================

dot-jax builds on several decades of work in biomedical optics, diffuse
optical tomography, and functional near-infrared spectroscopy.

Photon transport in tissue
--------------------------

Near-infrared light (650--950 nm) penetrates several centimetres into
biological tissue, scattered primarily by cell membranes and mitochondria,
and absorbed by hemoglobin, water, and lipids. The radiative transfer
equation (RTE) governs photon propagation, but the **diffusion
approximation** — valid when scattering dominates absorption — reduces it
to a tractable PDE:

.. math::

   -\nabla \cdot (D \, \nabla \Phi) + \mu_a \, \Phi = S

where :math:`\Phi` is the photon fluence rate,
:math:`D = 1/[3(\mu_a + \mu_s')]` is the diffusion coefficient,
:math:`\mu_a` is the absorption coefficient, and :math:`\mu_s'` is the
reduced scattering coefficient. This approximation was established by
Ishimaru (1978) and formalised for tissue optics by Patterson, Chance,
and Wilson (1989).

Analytical solutions
--------------------

Closed-form Green's functions exist for homogeneous geometries:

- **Infinite medium**: the point-source kernel
  :math:`\Phi(r) = \exp(-\mu_{\text{eff}} r) / (4\pi D r)`
- **Semi-infinite medium**: the image-source method with extrapolated
  boundary conditions (Farrell, Patterson & Wilson, 1992;
  Haskell et al., 1994)

These are implemented in :mod:`dot_jax.analytical` and remain important
for model validation.

Finite element methods for DOT
------------------------------

Real tissue geometries require numerical methods. The FEM approach was
introduced to biomedical optics by Arridge, Schweiger, Hiraoka & Delpy
(1993) and Paulsen & Jiang (1995). The diffusion equation maps naturally
to a symmetric positive-definite linear system:

.. math::

   (\mathbf{K} + \mathbf{M} + \mathbf{C}) \, \mathbf{\Phi} = \mathbf{b}

where **K** encodes diffusion, **M** encodes absorption, and **C**
enforces the Robin boundary condition.

The **TOAST** package (Schweiger & Arridge) and **NIRFAST** (Dehghani
et al., 2009) established the standard pipeline that redbirdpy and
dot-jax follow.

Chromophore spectroscopy
------------------------

The wavelength dependence of tissue absorption reveals chromophore
concentrations via the **modified Beer-Lambert law**. Hemoglobin
extinction data compiled by Scott Prahl at the Oregon Medical Laser
Center — tracing back to measurements by Takatani & Graham (1979)
and Zijlstra, Buursma & Meeuwsen-van der Roest (1991) — provides the
spectroscopic basis for fNIRS.

The **isosbestic point** near 800 nm, where oxy- and deoxyhemoglobin
have equal extinction, is a key design constraint for fNIRS instruments.
This is implemented in :mod:`dot_jax.property`.

Image reconstruction
--------------------

DOT image reconstruction is the inverse of the forward problem: given
boundary measurements, recover the spatial distribution of optical
properties. The standard approach (Arridge, 1999) uses:

1. **Linearisation** via the Born or Rytov approximation
2. The **adjoint Jacobian**:
   :math:`J_{d,s,n} = -\Phi_s(n) \cdot \Phi_d(n) \cdot V_n`
3. **Tikhonov regularisation** to stabilise the inversion

dot-jax implements this in :mod:`dot_jax.recon` and :mod:`dot_jax.spectral`,
with the additional capability of computing Jacobians via automatic
differentiation through the full forward solve.

The redbirdpy lineage
---------------------

dot-jax is a direct reimplementation of **redbirdpy** by Qianqian Fang,
which is a Python port of the MATLAB **redbird** toolbox. Fang's broader
ecosystem includes `MCX <http://mcx.space/>`_ for stochastic photon
transport and `iso2mesh <http://iso2mesh.sf.net/>`_ for tetrahedral mesh
generation.

dot-jax preserves the mathematical formulations and cross-validates against
redbirdpy at every layer, while adding automatic differentiation, JIT
compilation, and functional composition via JAX transformations.

Key references
--------------

1. A. Ishimaru, *Wave Propagation and Scattering in Random Media*,
   Academic Press, 1978.
2. F. F. Jöbsis, "Noninvasive, infrared monitoring of cerebral and
   myocardial oxygen sufficiency and circulatory parameters,"
   *Science*, 198(4323):1264--1267, 1977.
3. T. J. Farrell, M. S. Patterson, and B. C. Wilson, "A diffusion
   theory model of spatially resolved, steady-state diffuse reflectance,"
   *Med. Phys.*, 19(4):879--888, 1992.
4. S. R. Arridge, M. Schweiger, M. Hiraoka, and D. T. Delpy,
   "A finite element approach for modeling photon transport in tissue,"
   *Med. Phys.*, 20(2):299--309, 1993.
5. R. C. Haskell et al., "Boundary conditions for the diffusion
   equation in radiative transfer," *JOSA A*, 11(10):2727--2741, 1994.
6. S. R. Arridge, "Optical tomography in medical imaging,"
   *Inverse Problems*, 15(2):R41--R93, 1999.
7. S. Prahl, "Optical absorption of hemoglobin," Oregon Medical Laser
   Center, https://omlc.org/spectra/hemoglobin/, 1999.
8. D. A. Boas et al., "Imaging the body with diffuse optical
   tomography," *IEEE Signal Processing Magazine*, 18(6):57--75, 2001.
9. H. Dehghani et al., "Near infrared optical tomography using NIRFAST,"
   *Int. J. Numer. Methods Biomed. Eng.*, 25(6):711--732, 2009.
10. Q. Fang, "Mesh-based Monte Carlo method using fast ray-tracing in
    Plucker coordinates," *Biomed. Opt. Express*, 1(1):165--175, 2010.
