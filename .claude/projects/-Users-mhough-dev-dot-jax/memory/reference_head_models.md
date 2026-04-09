---
name: Standard head models survey
description: All publicly available head models for DOT/fNIRS/EEG/TMS — prioritized for dot-jax integration
type: reference
---

## Head model priority for dot-jax

### Tier 1 — Ready to load (tet FEM)
1. **Colin27 (Fang)**: 70K nodes, 423K tets, 4 layers, optical properties at 630nm. Public domain. JMesh/.mat. THE standard for DOT. Already partially working via fetch_neurojson.
2. **BrainMeshLibrary**: 5 layers, 35 age groups (2wk-89y) + 20 BrainWeb subjects. JMesh. DataLink API currently broken.
3. **UCL 4D Neonatal**: 9 layers, 16 weekly models (29-44 weeks PMA). On NeuroJSON.

### Tier 2 — Need Gmsh parser
4. **SimNIBS Ernie/MNI152**: 5-10 layers, ~3.5M tets. Gmsh v2 binary. `meshio` library reads this.
5. **SimNIBS Population**: 100 HCP subjects (22-35y). Gmsh format. Published Nat Sci Data March 2025.
6. **dHCP Database**: 215 individual neonatal models. .mat format.

### Tier 3 — Need meshing from segmentation
7. **MIDA (IT'IS)**: 115 tissues, 500um, most detailed. Voxel + STL. Free with license agreement.
8. **ICBM-NY / "Bikson model"**: 6 layers. Abaqus/COMSOL format. From CCNY (Parra + Bikson are same institution). ROAST automates this.
9. **SCI Head Model (Utah)**: 8 layers, T1/T2/DWI/fMRI/EEG. ~15.7M-60.2M tets. CC-BY. See separate memory.
10. **PHM**: 50 adult models. STL surfaces.

### Key finding
"Bikson head model" = ICBM-NY (New York Head). Both from CCNY. ROAST automates the Simpleware/Abaqus pipeline.

### The JAX advantage at scale
35 age models × 15 GB Jacobian each = 525 GB pre-computed. dot-jax: 0 bytes — compute on the fly from any mesh.

**How to apply:** Full details in paper/head_model_survey.md. Next step: implement Gmsh reader to unlock SimNIBS models (100+ subjects).
