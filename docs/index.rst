dot-jax
=======

**JAX/Equinox toolbox for Diffuse Optical Tomography (DOT) and fNIRS.**

A differentiable, GPU-accelerated reimplementation of `redbirdpy
<https://github.com/fangq/redbirdpy>`_ using the JAX ecosystem.
Enables autodiff Jacobians, JIT compilation, and vmap over
sources/detectors/wavelengths.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/analytical
   tutorials/optical_properties
   tutorials/fem_forward
   tutorials/reconstruction

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/analytical
   api/property
   api/mesh
   api/assembly
   api/forward
   api/spectral
   api/recon

.. toctree::
   :maxdepth: 1
   :caption: Background

   research

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
