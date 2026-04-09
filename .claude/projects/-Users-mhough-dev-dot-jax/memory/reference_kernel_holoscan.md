---
name: Kernel Flow + NVIDIA Holoscan BCI pipeline
description: Architecture of the Kernel/NVIDIA real-time fNIRS pipeline and how dot-jax complements it
type: reference
---

## Kernel Flow 2 + Holoscan BCI Visualization

**HoloHub source:** github.com/nvidia-holoscan/holohub/tree/main/applications/bci_visualization
**Sample data:** huggingface.co/datasets/KernelCo/holohub_bci_visualization (Apache-2.0)
**kernel-sdk:** PyPI binary wheels (NOT open source, no license declared, proprietary Rust bindings)
**Reconstruction code:** Apache-2.0 (Kernel copyright)

### Kernel Flow 2 specs
- 120 laser sources (690/905 nm), 240 SPAD detectors
- 6990 channels, 4.75 Hz
- Time-domain fNIRS (100 ps pulses, DTOF moments)
- Also has EEG

### Holoscan pipeline operators
1. StreamOperator — kernel-sdk live or SNIRF replay
2. BuildRHSOperator — log(0th moment), frame-to-frame diff, Jacobian mapping
3. NormalizeOperator — row normalization per moment/wavelength
4. RegularizedSolverOperator — Tikhonov dual solve: x = J^T (JJ^T + λI)^{-1} d
5. ConvertToVoxelsOperator — spectral unmix (Prahl extinction CSV) → HbO/HbR
6. VoxelStreamToVolumeOp — resample to anatomy NIfTI mask
7. VolumeRendererOp (ClaraViz) — GPU ray-casting

### Pre-computed files (NOT computed at runtime)
- `flow_mega_jacobian.npy` — 12-15 GB per device variant (5D: channels × moments × wavelengths × voxels × [mua,musp])
- `extinction_coefficients_mua.csv` — Prahl data (same as dot-jax)
- `flow_channel_map.json` — source-detector mapping

### What dot-jax provides that Holoscan doesn't
- Forward model (diffusion PDE → Green's functions → Jacobian)
- Automatic differentiation through the full solve
- On-the-fly Jacobian computation (no 15 GB files)
- TD-fNIRS forward model (Diffrax ODE)
- GCV/L-curve regularization selection
- Mesh flexibility (any head model, any age)

### What Holoscan provides that dot-jax doesn't
- Real-time streaming (<200ms/frame with cached Hessian)
- Sensor Bridge RDMA for hardware
- ClaraViz 3D volume rendering
- Production deployment on Jetson Thor / IGX

### Data downloaded
- `~/data/raw/kernel_flow/data.snirf` (496 MB, 1750 frames × 6990 ch)
- Tested: dot-jax reads it successfully, processes through full pipeline

**How to apply:** The two systems are complementary layers. dot-jax GENERATES the physics (Jacobians), Holoscan CONSUMES them at runtime. For the NeuroHack demo, dot-jax replays Kernel Flow SNIRF data with real-time reconstruction + web dashboard.
