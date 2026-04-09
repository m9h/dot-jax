---
name: NeuroJSON database resources for dot-jax
description: Available datasets from neurojson.io — head meshes, optical properties, fNIRS data, phantoms
type: reference
---

## NeuroJSON (neurojson.io) — by Qianqian Fang (MCX/redbirdpy creator)

API: `https://neurojson.io:7777/{database}/{document}`
Cache: `~/.cache/dot-jax/neurojson/`
Access via: `dot_jax.io.fetch_neurojson(database, document)`

### Key databases

**brainmeshlibrary** (319 docs): BrainWeb (20 subjects) + NDMRI (35 age groups, 2wk-89y). 5-layer tet meshes in JMesh format. NOTE: DataLink API for binary data currently returns 404 — mesh binary data not fetchable. Metadata/structure available.

**mcx** (20 docs): Monte Carlo benchmarks with optical properties. Colin27 (5-layer, 181x217x181), 4layer_head, usc19-5_atlas, sphshells, skinvessel, digimouse. `get_mcx_optical_properties("colin27")` WORKS — returns mua/mus/g/n per tissue.

**cotilab** (6 docs): CSF optical properties (2025), NeuroCaptain headcaps (30+ age groups), printable DOI phantoms (STL + recipes), MacDOT compressive DOT, TOBI breast DOT with SNIRF.

**openfnirs** (10 datasets): BIDS-fNIRS with SNIRF — auditory, tapping, motion artifact, synthetic HRF.

**bfnirs** (7 datasets): Boston fNIRS including BallSqueezingHD (HD-fNIRS, 12 subjects).

**ucl-4d-neonatal-head-model** (16 models): 29-44 weeks PMA, 9 tissue types, tet meshes + 10-5 positions.

**fnirs2mw** (1 dataset, 87 subjects): Frequency-domain fNIRS (amplitude + phase).

### Colin27 optical properties (at ~800 nm)

| Tissue | mua (1/mm) | musp (1/mm) | n |
|--------|-----------|------------|-----|
| scalp | 0.019 | 0.86 | 1.37 |
| skull | 0.019 | 0.86 | 1.37 |
| CSF | 0.0004 | 0.001 | 1.37 |
| GM | 0.020 | 0.99 | 1.37 |
| WM | 0.080 | 4.50 | 1.37 |

Cached at: `~/.cache/dot-jax/neurojson/mcx_colin27.json`
Volume saved: `~/.cache/dot-jax/neurojson/colin27_seg.npy` (181x217x181 uint8)

**How to apply:** Use `fetch_neurojson("mcx", "colin27")` for optical properties. The Colin27 volume can be meshed with dot-jax's marching cubes + Delaunay pipeline (tested: 3K-5K nodes, 4 pipeline variants, brain-only and full-head with tissue labels).
