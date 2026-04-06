# Definitive Survey: Standard Head Models for dot-jax

## Key Finding

**The "Bikson head model" IS the New York Head (ICBM-NY).** Marom Bikson
(CCNY) and Lucas Parra (CCNY) are at the same institution. The ICBM-NY
was built with Simpleware ScanIP + Abaqus FEM, and ROAST automates this
pipeline. There is no separate "Bikson model."

## Summary Table: All Available Head Models

| Model | Layers | Nodes/Elem | Tet FEM? | Optical? | Age | Format | License |
|-------|--------|-----------|----------|----------|-----|--------|---------|
| **Colin27 (Fang)** | 4 | 70K/423K | YES | **YES** | Adult | JMesh/.mat | Public domain |
| **BrainMeshLibrary** | 5 | varies | YES | No | 2wk–89y (35 ages) | JMesh | Open |
| **UCL 4D Neonatal** | 9 | varies | YES | No | 29–44 weeks | .mat/JMesh | Free |
| **dHCP Database** | 9 | varies | YES | No | 29–44 weeks (215 subjects) | .mat | Free |
| **SimNIBS Ernie** | 5–10 | ~3.5M tets | YES | No | Adult | Gmsh .msh | GPLv3 |
| **SimNIBS MNI152** | 5–10 | ~3.5M tets | YES | No | Adult template | Gmsh .msh | GPLv3 |
| **SimNIBS Population** | 5 | ~3.5M tets | YES | No | 22–35y (100 subj) | Gmsh .msh | Open |
| **ICBM-NY (=Bikson)** | 6 | ~millions | YES | No | Adult template | Abaqus/COMSOL | Free |
| **MIDA** | 115 | voxel (mesh by user) | User mesh | No* | Adult | .nii/STL | Free† |
| **ViP (IT'IS)** | 300+ | voxel | User mesh | No* | 6–37y (8 models) | Sim4Life | Commercial |
| **PHM** | 7+ | surface | User mesh | No | 22–35y (50 subj) | SAT/STL | Free† |
| **IXI Heads** | 60 | voxel | User mesh | No* | 29–67y (4 subj) | NIfTI | Free† |
| **FieldTrip BEM** | 3 | surface | NO (BEM) | No | Adult (Colin27) | .mat | GPLv3 |
| **MNE fsaverage** | 3 | surface | NO (BEM) | No | Adult (40-subj avg) | FreeSurfer | BSD |
| **BrainStorm** | 3–5 | surface/tet | Via iso2mesh | No | Adult (ICBM152) | .mat | GPLv3 |
| **MCX benchmarks** | 4–6 | voxel | N/A (MC) | **YES** | Adult (Colin27) | JSON | CC-BY |

\* IT'IS tissue properties database available separately
† Handling fees / license agreement required

## Priority for dot-jax Integration

### Tier 1: Ready to load NOW (tetrahedral FEM with node/elem arrays)

1. **Colin27 (Fang)** — 4 layers, optical properties at 630 nm, public domain.
   Already loadable via `read_jmesh()` / MCX `fetch_neurojson()`.

2. **BrainMeshLibrary** — 5 layers, 35 age groups + 20 BrainWeb subjects.
   JMesh format (DataLink API currently down; GitHub download works).

3. **UCL 4D Neonatal** — 9 layers, 16 weekly models (29–44 weeks).
   On NeuroJSON. Critical for infant fNIRS.

### Tier 2: Need a mesh format parser

4. **SimNIBS models** (Ernie, MNI152, 100 HCP population) — Gmsh v2 binary
   format. The `meshio` Python library reads this. Adding a `read_gmsh()`
   to `io.py` unlocks 100+ models instantly.

5. **dHCP Database** — 215 individual neonatal models. MATLAB .mat format
   with node/elem arrays (like Colin27).

### Tier 3: Need meshing from segmentation

6. **MIDA** — 115-tissue voxel model. Requires iso2mesh/CGAL tetrahedralization.
   Most anatomically detailed model available.

7. **ICBM-NY** — 6-layer. Abaqus .inp format (parseable but non-trivial).

8. **PHM** — 50 adult models. STL surfaces need tet meshing.

## The JAX Advantage with Multiple Models

**This is where dot-jax's differentiability pays off at scale:**

With pre-computed (static) Jacobians, each head model requires a separate
15 GB Jacobian file. For 35 age groups × 2 wavelengths = 70 Jacobians =
**~1 TB of pre-computed files.**

With dot-jax, the forward model computes the Jacobian on-the-fly from the
mesh. Load a different head model → `FEMMesh.create(node, elem)` → new
Jacobian in 30 seconds. **Zero pre-computation, zero storage.**

```python
# Process an infant with the 36-week neonatal model
mesh_36w, labels = load_brain_mesh("UCL4D", "36weeks")
pipe = RealtimePipeline(mesh_36w, srcpos, detpos, mua=0.02, musp=0.99)

# Process an elderly subject with the 80-84y model
mesh_80y, labels = load_brain_mesh("NDMRI", "80-84Years")
pipe = RealtimePipeline(mesh_80y, srcpos, detpos, mua=0.02, musp=0.99)

# Same code, different mesh. Jacobian computed automatically.
# jax.grad works on both. No pre-computation needed.
```

For a population study across the lifespan (2 weeks to 89 years), the
Kernel/Holoscan approach would need 35 × 15 GB = **525 GB** of pre-computed
Jacobians. dot-jax needs **0 bytes** — just the meshes (~10 MB each).

## Software for Generating Custom Models

| Tool | Input | Output | Best for |
|------|-------|--------|----------|
| **SimNIBS charm** | T1 MRI (+T2 optional) | 10-tissue tet mesh | Subject-specific from MRI |
| **ROAST** | T1 MRI (or MNI152) | 5-tissue tet mesh | Quick tDCS/TMS modeling |
| **brain2mesh** | SPM/FreeSurfer seg | 5-layer tet mesh | Batch processing |
| **iso2mesh** | Binary/gray volumes | Tet mesh | General-purpose meshing |
| **dot-jax atlas** | MNI152 tissue probs | Tet mesh (FEMMesh) | No external deps needed |

## Key References

1. Huang Y et al. "The New York Head." NeuroImage 140:150-162, 2016.
2. Thielscher A et al. SimNIBS. IEEE EMBC 2015.
3. Puonti O et al. "charm segmentation." NeuroImage 219:117044, 2020.
4. SimNIBS Population. Nat Sci Data, March 2025.
5. Iacono MI et al. "MIDA Model." PLoS ONE 10(4):e0124126, 2015.
6. Gosselin MC et al. "Virtual Population." Phys Med Biol 59(18):5287, 2014.
7. Brigadoi S et al. "UCL 4D Neonatal." NeuroImage 100:385-394, 2014.
8. Collins-Jones LH et al. "dHCP Head Models." Hum Brain Mapp 42(3):567, 2021.
9. Tran AP et al. "Mesh-based fNIRS." Neurophotonics 7(1):015008, 2020.
10. Fang Q, Boas D. "Tet mesh generation." IEEE ISBI 2009.
