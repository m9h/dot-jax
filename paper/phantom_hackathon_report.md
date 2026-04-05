# 3D-Printable Optical Phantom Hackathon Report

## Overview

This report covers the requirements for building 3D-printed optical phantoms for
validating the dot-jax DOT/fNIRS pipeline, based on the COTILAB
`Printable_DOI_Phantoms_2025` dataset from NeuroJSON and published literature.

## 1. Target Optical Properties

From the MCX Colin27 benchmark and dot-jax defaults, the tissue optical
properties we need to match (at ~800 nm NIR):

| Tissue | mua (1/mm) | musp (1/mm) | n | Notes |
|--------|-----------|------------|-----|-------|
| Scalp | 0.019 | 0.86 | 1.37 | Similar to skull |
| Skull | 0.019 | 0.86 | 1.37 | Bone-like |
| CSF | 0.0004 | 0.001 | 1.37 | Nearly transparent |
| Gray matter | 0.020 | 0.99 | 1.37 | Primary DOT target |
| White matter | 0.080 | 4.50 | 1.37 | Highly scattering |

### dot-jax chromophore values at key wavelengths

From `property.py` built-in tables (Prahl/OMLC):

| Wavelength | HbO2 ext | Hb ext | Water mua |
|------------|----------|--------|-----------|
| 690 nm | 6.36e-5 | 5.38e-4 | 5.44e-4 |
| 750 nm | 1.19e-4 | 3.24e-4 | 2.60e-3 |
| 800 nm | 1.88e-4 | 1.75e-4 | 2.00e-3 |
| 830 nm | 2.24e-4 | 1.60e-4 | 3.38e-3 |
| 850 nm | 2.44e-4 | 1.59e-4 | 4.30e-3 |

Units: HbO2/Hb in 1/(mm*uM), water in 1/mm.

## 2. COTILAB Printable Phantoms (NeuroJSON)

The `Printable_DOI_Phantoms_2025` dataset from Fang's COTILAB contains:

### Available STL geometries
- **Validation blocks**: Rectangular slabs with known dimensions for calibration
- **Digimouse body/brain**: Anatomically realistic mouse phantoms
- **Titration rulers**: Graduated thickness samples for systematic property variation
- **Multi-layer head phantoms**: Simplified head geometry with tissue layers

### Reported property targets and measurements
The phantoms use **resin-based 3D printing** with:
- **TiO2 nanoparticles** for scattering (controls musp)
- **India ink or nigrosin** for absorption (controls mua)
- **Clear resin** as the base matrix (n ≈ 1.5)

Typical recipes achieve:
- mua range: 0.001 -- 0.1 /mm (covers all tissue types)
- musp range: 0.5 -- 5.0 /mm (covers scalp through white matter)

## 3. Required 3D Printing Facilities

### Recommended: SLA/DLP Resin Printer

**Why resin, not FDM:**
- FDM (filament) printers create layer lines that act as scattering artifacts
- Resin printers produce optically smooth, homogeneous volumes
- Resin can be doped with precise concentrations of scatterers/absorbers
- Published validation shows resin phantoms match target properties within 5-10%

**Recommended printers:**

| Printer | Type | Resolution | Build volume | Cost |
|---------|------|-----------|-------------|------|
| Formlabs Form 3+ | SLA | 25 um | 145x145x185 mm | ~$3,500 |
| Elegoo Saturn 3 | MSLA | 28 um | 218x123x250 mm | ~$500 |
| Anycubic Photon M5s | MSLA | 19 um | 218x123x200 mm | ~$400 |
| Prusa SL1S Speed | MSLA | 47 um | 127x80x150 mm | ~$1,500 |

**Budget option:** Elegoo Saturn 3 (~$500) provides excellent resolution for
optical phantoms at a fraction of the cost of Formlabs.

### Materials

| Material | Purpose | Source | Notes |
|----------|---------|--------|-------|
| Clear UV resin | Base matrix | Standard printer resin | n ≈ 1.5 at NIR |
| TiO2 nanoparticles | Scattering agent | Sigma-Aldrich (rutile, <100 nm) | ~0.1-2 mg/mL for tissue-like musp |
| India ink | Absorber | Art supply store | ~0.01-0.5 uL/mL for tissue-like mua |
| Nigrosin dye | Absorber (alternative) | Sigma-Aldrich | More stable than ink, better NIR absorption |
| White pigment paste | Combined scatterer | Printer resin additive | Pre-dispersed TiO2 |

### Mixing protocol (from literature)

1. **Base resin**: Start with clear photopolymer resin
2. **Scattering**: Add TiO2 powder, sonicate 30 min to disperse
   - ~0.5 mg/mL TiO2 → musp ≈ 1.0 /mm (gray matter-like)
   - ~2.0 mg/mL TiO2 → musp ≈ 4.0 /mm (white matter-like)
3. **Absorption**: Add India ink from diluted stock
   - ~0.05 uL/mL ink → mua ≈ 0.02 /mm (gray matter-like)
   - ~0.2 uL/mL ink → mua ≈ 0.08 /mm (white matter-like)
4. **Mix thoroughly**, degas in vacuum chamber
5. **Print** in standard resin printer workflow
6. **Post-cure** under UV (per resin manufacturer specs)

## 4. Validation Workflow

### Measuring phantom optical properties

1. **Integrating sphere + spectrophotometer** (gold standard)
   - Measure total reflectance and transmittance
   - Inverse Adding-Doubling (IAD) algorithm to extract mua, musp
   - Equipment: ~$10K for a basic setup

2. **Spatial frequency domain imaging (SFDI)** (non-contact)
   - Project sinusoidal patterns, capture diffuse reflectance
   - Extract mua, musp from spatial frequency response
   - Can map heterogeneous phantoms

3. **Time-domain spectroscopy** (most accurate)
   - Pulsed laser + TCSPC detector
   - Fit temporal point spread function for mua, musp
   - Equipment: ~$50K+

4. **dot-jax forward model comparison** (computational)
   - Model the phantom geometry in dot-jax
   - Compare predicted vs measured detector values
   - Iterate on properties to best-fit

### Recommended for hackathon: SFDI or dot-jax computational validation

## 5. Hackathon Logistics

### What to prepare in advance (1-2 weeks before)
- [ ] Print phantom geometries (4-6 hour print time each)
- [ ] Prepare 3-4 resin batches with different optical properties
- [ ] Calibrate with reference samples (known mua/musp)
- [ ] Print titration ruler for property gradient validation
- [ ] Prepare source-detector arrays (fiber bundles or direct LED/photodiode)

### What can be done live at hackathon
- [ ] Load phantom geometry into dot-jax as FEMMesh
- [ ] Assign measured optical properties to mesh regions
- [ ] Run forward model predictions
- [ ] Acquire measurements on physical phantom
- [ ] Compare predicted vs measured detector values
- [ ] Iterate: adjust properties, re-run model, compare
- [ ] Demonstrate reconstruction on phantom with inclusion

### Suggested hackathon phantom designs

1. **Homogeneous slab** (50x50x30 mm)
   - mua = 0.02, musp = 1.0 (brain-like)
   - Simplest geometry, validates basic forward model

2. **Two-layer slab** (50x50x30 mm, 5mm "scalp" + 25mm "brain")
   - Layer 1: mua = 0.019, musp = 0.86 (scalp)
   - Layer 2: mua = 0.02, musp = 0.99 (gray matter)
   - Tests boundary condition handling

3. **Slab with inclusion** (50x50x30 mm + 10mm sphere)
   - Background: mua = 0.01, musp = 1.0
   - Inclusion: mua = 0.05, musp = 1.0 (2.5x absorption contrast)
   - Tests image reconstruction

4. **Simplified head phantom** (hemisphere, 80mm diameter)
   - 3 layers: scalp (5mm), skull (7mm), brain
   - Most realistic, tests multi-tissue forward model

### Cost estimate

| Item | Cost |
|------|------|
| Resin printer (Elegoo Saturn 3) | $500 |
| Clear resin (1L) | $30-50 |
| TiO2 nanoparticles (100g) | $30-50 |
| India ink | $5-10 |
| Nigrosin dye (25g) | $30 |
| Ultrasonic bath (for TiO2 dispersion) | $50-100 |
| Source-detector fiber optics (basic) | $200-500 |
| **Total (basic setup)** | **~$850-1,250** |

With existing access to a resin printer, the materials cost drops to ~$100-200.

## 6. Key References

1. Dempsey et al. (2017) "Geometrically complex 3D-printed phantoms for
   diffuse optical imaging" — Biomed. Opt. Express
2. Dong et al. (2021) "3D printed tissue-simulating phantoms for near-infrared
   spectroscopy" — J. Biomed. Opt.
3. Fang et al. (2025) "Printable DOI Phantoms" — COTILAB/NeuroJSON dataset
4. Pogue & Patterson (2006) "Review of tissue simulating phantoms for optical
   spectroscopy, imaging and dosimetry" — J. Biomed. Opt.
5. Ntziachristos et al. (2001) "Fluorescence molecular tomography resolves
   protease activity in vivo" — Nature Medicine (phantom validation section)
