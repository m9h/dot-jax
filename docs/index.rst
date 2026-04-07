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
   tutorials/hddot_processing
   tutorials/kernel_flow
   tutorials/realtime
   tutorials/fnirs_pipeline
   tutorials/forward_modeling
   tutorials/image_reconstruction
   tutorials/analytical_solutions

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/types
   api/analytical
   api/property
   api/mesh
   api/assembly
   api/forward
   api/spectral
   api/recon
   api/atlas
   api/hemodynamics
   api/io
   api/td_forward
   api/realtime
   api/streaming

.. toctree::
   :maxdepth: 1
   :caption: Background

   research

.. toctree::
   :maxdepth: 1
   :caption: Project

   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
