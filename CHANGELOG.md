# Changelog

All notable changes to dot-jax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-04-05

### Added

- **Core forward pipeline**: analytical solutions (infinite/semi-infinite CW
  fluence), chromophore extinction spectra and Beer-Lambert law, FEMMesh
  Equinox module, FEM system matrix assembly (K + M + C), CW forward solver
  via Lineax, multi-wavelength spectral forward model, and Gauss-Newton
  image reconstruction with Tikhonov/LM regularisation.
- **Sparse forward solver** (`forward_cw_sparse`): iterative CG solver via
  BCSR sparse matrices.
- **I/O module**: SNIRF reader, BIDS-fNIRS loader, JMesh/NeuroJSON reader,
  and MCX optical property database access.
- **Atlas module**: MNI152 atlas head mesh generation and optode projection
  for high-density DOT arrays.
- **Hemodynamics module**: intensity-to-OD, bandpass filtering, downsampling,
  spectral unmixing, channel SNR/pruning, motion artifact detection and
  spline correction, short-channel regression, z-score normalisation, and
  global variance of temporal derivatives (GVTD).
- **Time-domain forward model** via Diffrax: mass matrix assembly for
  temporal discretisation, source pulse generation, and DTOF moment
  computation.
- **Reconstruction enhancements**: L-curve and GCV regularisation parameter
  selection, dual-formulation solver.
- **Real-time pipeline**: pre-computed inverse operators, epoch accumulator,
  and streaming inference with WebSocket/HTTP dashboard.
- **Examples 01--04**: analytical solutions, optical properties, FEM forward
  solve, and image reconstruction tutorials.
- **Examples 05--07**: HD-DOT processing pipeline (ds004569), Kernel Flow 2
  TD-fNIRS demo, and real-time fNIRS dashboard for Global NeuroHack.
- **Sphinx documentation** with ReadTheDocs configuration, API reference for
  all modules, and research background.
- **169 tests** covering mathematical properties, known-value validation,
  cross-validation against redbirdpy/scipy, and JIT/grad/vmap compatibility.
