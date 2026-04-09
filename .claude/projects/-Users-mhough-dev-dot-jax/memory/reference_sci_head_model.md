---
name: SCI Head Model (Utah) details
description: Multimodal head model from Scientific Computing and Imaging Institute — T1/T2/DWI/fMRI/EEG + 8-layer tet mesh
type: reference
---

## SCI Head Model

**Source:** Scientific Computing and Imaging (SCI) Institute, University of Utah
**URL:** https://sci.utah.edu/sci-headmodel/
**Paper:** Warner A, Tate J, Burton B, Johnson CR. "A High-Resolution Head and Brain Computer Model." bioRxiv 2019. doi:10.1101/552190
**License:** CC-BY
**Subject:** Single healthy adult female, age 23

### Modalities
- T1w MRI (1mm iso, DICOM + NRRD)
- T2w MRI (DICOM + NRRD)
- Diffusion MRI (AP+PA phase encoding, FSL topup corrected)
- DTI (eigenvalues + eigenvectors via FSL DTIFIT)
- fMRI (DICOM + corrected NIfTI)
- Pseudo-CT (MR-based synthetic CT)
- 128-channel EEG (.edf + .mat)
- 256-channel EEG (.edf + .mat)

### Segmentation: 8 tissue layers
1. White matter
2. Gray matter
3. CSF
4. Skull
5. Scalp
6. Sinus / air cavities
7. Eyes
8. Background / air

### Mesh
- Created with Seg3D (segmentation) + Cleaver (tet meshing)
- Two resolutions: ~60.2M elements (high-res) and ~15.7M elements (lower-res)
- Formats: .ele, .elem, .node, .ply, .vtk
- Includes SCIRun networks for FEM forward simulation
- Supports isotropic AND anisotropic (DTI-derived) conductivity

### Why it matters for dot-jax
- **Multimodal:** The DWI/DTI data can constrain optical scattering anisotropy (white matter scatters more along fiber direction)
- **Cross-tool:** DWI → sbi4dwi, fMRI → neurojax, EEG → lead field, fNIRS → dot-jax — all from same subject
- **TRIBE integration:** The DTI-derived structural connectivity can feed the TRIBE model to predict fNIRS activation
- **8-layer mesh:** More tissue detail than Colin27 (4 layers) or BrainMeshLibrary (5 layers)

### Limitations
- NOT on NeuroJSON (would need manual download)
- Very large meshes (15.7M-60.2M tets) — dot-jax's dense solver handles ~10K nodes; sparse solver ~100K. Would need decimation or the sparse CG path.
- No optical properties included — would need to assign from MCX Colin27 values
- Mesh formats (.node/.elem/.vtk) need parser (meshio handles .vtk)
- Only one subject (not a population atlas)

**How to apply:** Download from sci.utah.edu, load .vtk mesh via meshio, assign optical properties from MCX Colin27 table, decimate to 5K-10K nodes for dot-jax. The DWI data feeds sbi4dwi for TRIBE-based fNIRS prediction.
