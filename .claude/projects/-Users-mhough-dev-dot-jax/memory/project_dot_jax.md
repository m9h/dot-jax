---
name: dot-jax project state and architecture
description: Complete project state for dot-jax DOT/fNIRS JAX toolbox — modules, tests, data, demos, hackathon prep
type: project
---

## dot-jax: JAX/Equinox toolbox for Diffuse Optical Tomography and fNIRS

**Repo:** /Users/mhough/dev/dot-jax (github.com/m9h/dot-jax, GPL-3.0)
**Tests:** 325+ passing across 13 test files
**Python:** 3.11-3.13, uses `uv` for package management (NOT pip)

### Module inventory (src/dot_jax/)

| Module | Lines | Status | Key functions |
|--------|-------|--------|---------------|
| `_types.py` | 22 | Complete | C0, R_C0, ForwardResult, ReconResult |
| `analytical.py` | ~300 | Complete | getreff, infinite_cw, semi_infinite_cw, spbesselj/y/h, spharmonic |
| `property.py` | ~314 | Complete | extinction (Prahl/OMLC), mua_from_concentrations, musp_from_scattering, musp2sasp |
| `mesh.py` | ~350 | Complete | FEMMesh (Equinox Module), compute_evol/face_area/deldotdel/nvol, elem2node, extract_surface, reorient_elems, smooth_on_mesh |
| `assembly.py` | ~178 | Complete | assemble_stiffness (K), assemble_mass (M), assemble_boundary (C), assemble_system_cw (A=K+M+C) |
| `forward.py` | ~241 | Complete | locate_sources, assemble_rhs, get_detector_values, forward_cw (dense LU), forward_cw_sparse (BCSR+CG) |
| `spectral.py` | ~140 | Complete | spectral_forward_cw, compute_jacobian_mua, normalize_jacobian |
| `recon.py` | ~250 | Complete | reconstruct_mua (Gauss-Newton), reconstruct_image (depth-weighted), solve_dual (Kernel-style), compute_lcurve, select_lambda_lcurve, select_lambda_gcv |
| `hemodynamics.py` | ~400 | Complete | intensity_to_od, bandpass_filter, downsample, spectral_unmix, compute_channel_snr, prune_channels, detect_motion_artifacts, correct_motion_spline, normalize_od, zscore_images, compute_gvtd, identify_short_channels, regress_short_channels |
| `io.py` | ~650 | Complete | read_snirf, load_bids_nirs, snirf_to_dot_jax, read_jmesh, fetch_neurojson, get_mcx_optical_properties, load_brain_mesh |
| `atlas.py` | ~213 | Complete | fetch_mni152_seg, generate_head_mesh, project_to_surface |
| `td_forward.py` | ~170 | Complete | assemble_mass_time, td_source_pulse, td_forward_cw (Diffrax ODE), dtof_moments |
| `realtime.py` | ~200 | Complete | RealtimePipeline (pre-computed Jinv, <100ms/frame), EpochAccumulator |
| `streaming.py` | ~250 | Complete | FramePacket, SNIRFReplayStreamer, ZMQFrameReceiver, DashboardServer |

### Data downloaded

- `~/data/raw/ds004569/` — OpenNeuro HD-DOT dataset (Sherafati et al. 2025)
  - sub-01/ses-01 (513 MB) + sub-02 all 7 sessions (3.8 GB)
  - 96 src, 92 det, 3684 ch, 750/850 nm, 10 Hz
- `~/data/raw/kernel_flow/data.snirf` — Kernel Flow 2 sample (496 MB, HuggingFace KernelCo)
  - 105 src, 210 det, 6990 ch, 690/905 nm, 4.75 Hz
- `~/.cache/dot-jax/neurojson/` — Colin27 volume + optical properties from NeuroJSON MCX

### Demos (examples/)

1. `01_analytical_solutions.py` — CW fluence, semi-infinite, autodiff
2. `02_optical_properties.py` — chromophore spectra, Beer-Lambert
3. `03_fem_forward_solve.py` — mesh, system matrix, Jacobian
4. `04_image_reconstruction.py` — synthetic data, Gauss-Newton
5. `05_hddot_processing.py` — full ds004569 pipeline (12 steps)
6. `06_kernel_flow_demo.py` — Kernel Flow 2 end-to-end with dual solver
7. `07_realtime_demo.py` — streaming dashboard (replay or live ZMQ)

### Papers and docs (paper/)

- `dot_jax.Rnw` / `dot_jax.pdf` — research paper (Sweave/knitr, 7 pages)
- `references.bib` — 25 references
- `neurohack_protocol.md` — fNIRS-guided SAINT TMS targeting protocol
- `phantom_hackathon_report.md` — 3D printing guide for optical phantoms
- `head_model_survey.md` — definitive survey of all standard head models

### Sphinx docs (docs/)

- Built with RTD theme, numpydoc, autodoc, intersphinx
- `.readthedocs.yaml` configured
- Build: `Rscript -e "knitr::knit('dot_jax.Rnw')" && tectonic dot_jax.tex`

### Key architectural points

- Everything is differentiable via jax.grad (including TD-fNIRS ODE solve via Diffrax)
- Cross-validated against redbirdpy at every layer
- The Kernel/NVIDIA Holoscan BCI app (Apache-2.0) loads pre-computed 15 GB Jacobians; dot-jax computes them on the fly from the diffusion PDE
- kernel-sdk is NOT open source (proprietary binary wheels, no license declared)
- The "Bikson head model" = ICBM-NY (New York Head), both from CCNY

### Global NeuroHack (April 10-12, 2026) prep

- Real-time pipeline ready (<100ms/frame with pre-computed Jinv)
- ZMQ streaming + web dashboard
- N-back task for DLPFC activation → SAINT TMS targeting
- TRIBE model integration concept for structural prior
- Kernel Flow sample data tested end-to-end

**Why:** dot-jax replaces 15 GB of static Jacobian files with 3000 lines of differentiable code. Any mesh, any optical properties, any wavelength — computed on the fly with jax.grad through the full forward solve.

**How to apply:** When continuing development, check this memory + the head_model_survey.md + neurohack_protocol.md for context. The user prefers red-green TDD and uv for Python.
