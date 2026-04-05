# NeuroHack Protocol: fNIRS-Guided TMS Target Identification with Kernel Flow 2 + dot-jax

## Global NeuroHack, April 10–12 2026

### Overview

This protocol uses **Kernel Flow 2** (TD-fNIRS/EEG) to identify the optimal
left dorsolateral prefrontal cortex (DLPFC) stimulation target for the **SAINT
protocol** (Stanford Accelerated Intelligent Neuromodulation Therapy). The
conventional SAINT approach requires an fMRI session to map DLPFC-sgACC
functional connectivity. We replace fMRI with real-time fNIRS tomographic
imaging via **dot-jax**, enabling portable, accessible, and immediate TMS
target identification.

### Clinical motivation

The SAINT protocol (Cole et al., 2020; 2022) delivers high-dose intermittent
theta-burst stimulation (iTBS) to the left DLPFC for treatment-resistant
depression, achieving ~80% remission rates in open-label trials. The critical
step is identifying the DLPFC subregion most functionally anticorrelated with
the subgenual anterior cingulate cortex (sgACC) — the target that predicts
clinical response (Fox et al., 2012). Current practice requires:

1. A structural MRI for neuronavigation
2. A resting-state fMRI for DLPFC-sgACC functional connectivity mapping
3. Offline analysis to identify the optimal target coordinate
4. Registration of the target to the neuronavigation system

**Our innovation:** Replace steps 2–3 with a 10-minute fNIRS session using
Kernel Flow 2, processed in real-time by dot-jax on a DGX Spark. The DLPFC
activation map from a working memory task identifies the functionally relevant
cortical subregion directly, without MRI.

---

## 1. Equipment

### 1.1 Kernel Flow 2 headset
- 120 laser sources (690/905 nm), 240 single-photon detectors
- 3000+ measurement channels, 4.75 Hz frame rate
- Time-domain fNIRS (100 ps pulses, DTOF moments)
- Integrated EEG electrodes
- Covers: frontal (F3/F4/Fz), temporal, parietal, occipital cortex
- **Key:** The frontal modules cover the DLPFC target region (F3/F5 in 10-20 system)

### 1.2 Stimulus presentation
**Primary recommendation: PsychoPy** (standalone)

PsychoPy provides the most reliable timing for fNIRS block designs and sends
event markers via:
- LSL (Lab Streaming Layer) — preferred for multi-device sync
- Parallel port TTL
- Serial port
- Network triggers (TCP/UDP)

**Alternative: jsPsych** (browser-based)
- Zero-install, runs in any browser
- Excellent for hackathon settings
- Can send markers via WebSocket to the dot-jax dashboard

**Alternative: EEG-ExPy** (NeuroTechX)
- Built-in paradigms: P300, N170, SSVEP, SSAEP, mental imagery
- **Does not include N-back** — would need custom implementation
- Kernel Flow support via BrainFlow backend (John Griffiths' contributions)
- PsychoPy backend, LSL event markers
- Good for EEG+fNIRS multimodal experiments
- For this protocol, **PsychoPy standalone is preferred** since we need
  a custom N-back task that EEG-ExPy does not provide out of the box

### 1.3 Computing
- **Acquisition laptop:** Runs kernel-sdk + PsychoPy, connected to headset
- **DGX Spark:** Runs dot-jax real-time pipeline + web dashboard
- **Network:** Ethernet between laptop and DGX (ZMQ frame streaming)

### 1.4 Optional: TMS system
- Any MagStim or MagVenture system with figure-8 coil
- Neuronavigation (Brainsight, Localite, or ANT) for target registration
- Note: Kernel Flow should be removed during TMS delivery (metallic components)

---

## 2. Experiment Paradigm: N-back Working Memory Task

The N-back task is the gold standard for DLPFC activation in fNIRS:

### 2.1 Task design (block design for fNIRS)

```
REST → 0-back → REST → 2-back → REST → 0-back → REST → 2-back → REST → ...
 30s     30s     30s     30s     30s     30s     30s     30s     30s
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Block duration | 30 s | Matches hemodynamic response (~6s rise, ~15s peak) |
| Rest duration | 30 s | Allow HbO to return to baseline |
| Conditions | 0-back (control), 2-back (DLPFC activation) | |
| Blocks per condition | 5 | Sufficient for reliable contrast |
| Total duration | 10 min (300 s) | 10 blocks × 30s task + 30s rest |
| Stimulus rate | 1 per 2.5 s (500 ms display, 2000 ms ISI) | |
| Stimuli | Single letters (A-Z) | Visual, central fixation |
| Target probability | 33% (1 in 3 stimuli are targets) | |

### 2.2 Conditions

**0-back (control):** "Press button when you see the letter X."
- Engages visual processing and motor response but minimal working memory
- Serves as the low-level control for DLPFC contrast

**2-back (DLPFC activation):** "Press button when the current letter matches
the letter shown 2 items ago."
- Engages working memory maintenance and updating
- Robustly activates bilateral DLPFC, with left lateralisation for verbal stimuli

### 2.3 Alternative DLPFC tasks

If N-back is problematic (e.g., subject comprehension issues), these
alternatives also reliably activate left DLPFC in fNIRS:

| Task | DLPFC activation | Ease of implementation |
|------|-----------------|----------------------|
| **Verbal/phonemic fluency** | Strong left DLPFC, most replicated in fNIRS | Very easy (say words starting with F/A/S) |
| **Stroop** (colour-word) | Bilateral DLPFC | Easy (PsychoPy built-in) |
| **Go/No-Go** | Right > left DLPFC | Easy |
| **Tower of London** | Bilateral, planning | Moderate complexity |

For purely left DLPFC lateralisation, **verbal fluency** is the best
alternative to N-back (Ehlis et al., 2009; Herrmann et al., 2005).

### 2.4 Expected fNIRS signal

Based on published fNIRS N-back studies (Herff et al., 2014; Fishburn
et al., 2014; Sato et al., 2013):

- **HbO increase** in DLPFC during 2-back vs 0-back: ~0.3–0.8 μM
- **HbR decrease:** ~0.1–0.3 μM (inverse of HbO, confirming neurovascular coupling)
- **Onset latency:** ~2–3 s after block onset (hemodynamic delay)
- **Peak latency:** ~6–8 s into task block
- **Return to baseline:** ~10–15 s after block offset
- **Spatial extent:** Channels over F3/F5 (left DLPFC) and F4/F6 (right DLPFC)
- **Lateralisation:** Left > right for verbal N-back
- **Reliability:** Test-retest ICC ~0.6–0.8 for DLPFC HbO during N-back
  (Plichta et al., 2006)

### 2.4 PsychoPy implementation

```python
#!/usr/bin/env python
"""N-back task for DLPFC activation mapping with Kernel Flow."""

from psychopy import visual, core, event, data
import random
import string

# --- Configuration ---
N_BLOCKS = 10  # 5 per condition
BLOCK_DURATION = 30.0  # seconds
REST_DURATION = 30.0
STIMULUS_DURATION = 0.5
ISI = 2.0  # inter-stimulus interval
TARGET_PROB = 0.33

# --- Window ---
win = visual.Window(fullscr=True, color='black')
text_stim = visual.TextStim(win, height=0.15, color='white')
fixation = visual.TextStim(win, text='+', height=0.1, color='white')
instruction = visual.TextStim(win, height=0.06, color='white', wrapWidth=1.5)

# --- Event marker (LSL or ZMQ) ---
# Option A: LSL
# from pylsl import StreamInfo, StreamOutlet
# info = StreamInfo('Markers', 'Markers', 1, 0, 'string', 'nback_markers')
# outlet = StreamOutlet(info)
# def send_marker(label): outlet.push_sample([label])

# Option B: ZMQ to dot-jax
import zmq
ctx = zmq.Context()
marker_sock = ctx.socket(zmq.PUB)
marker_sock.bind("tcp://*:15556")
def send_marker(label):
    marker_sock.send_string(label)

# --- Generate block sequence ---
conditions = ['0back', '2back'] * (N_BLOCKS // 2)
random.shuffle(conditions)

# --- Run ---
clock = core.Clock()

for block_idx, condition in enumerate(conditions):
    # Rest period
    send_marker(f'rest_start')
    fixation.draw()
    win.flip()
    core.wait(REST_DURATION)

    # Task block
    send_marker(f'{condition}_start')
    n = 0 if condition == '0back' else 2

    # Generate stimulus sequence
    letters = []
    n_stim = int(BLOCK_DURATION / (STIMULUS_DURATION + ISI))
    for i in range(n_stim):
        if i >= n and random.random() < TARGET_PROB:
            letters.append(letters[i - n])  # target
        else:
            letters.append(random.choice(string.ascii_uppercase))

    # Present stimuli
    for i, letter in enumerate(letters):
        is_target = (i >= n and letter == letters[i - n]) if n > 0 else (letter == 'X')

        text_stim.text = letter
        text_stim.draw()
        win.flip()
        send_marker(f'stim_{letter}_{"target" if is_target else "nontarget"}')
        core.wait(STIMULUS_DURATION)

        fixation.draw()
        win.flip()
        core.wait(ISI)

        # Check for response
        keys = event.getKeys(['space'], timeStamped=clock)
        if keys:
            send_marker('response')

    send_marker(f'{condition}_end')

# Final rest
send_marker('rest_start')
fixation.draw()
win.flip()
core.wait(REST_DURATION)
send_marker('experiment_end')

win.close()
core.quit()
```

### 2.5 Alternative: jsPsych (zero-install, browser-based)

For hackathon settings where PsychoPy installation is difficult:

```html
<!-- Save as nback.html, open in browser -->
<script src="https://unpkg.com/jspsych@7.3/dist/jspsych.js"></script>
<script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@1.1/dist/index.js"></script>
<!-- ... N-back implementation ... -->
<!-- Markers sent via WebSocket to dot-jax dashboard -->
```

---

## 3. Data Acquisition Protocol

### 3.1 Setup (10 min)

1. **Fit Kernel Flow 2** on subject's head
   - Align Cz reference to vertex (midpoint nasion–inion)
   - Ensure frontal modules cover F3/F5 region (left DLPFC target)
   - Check signal quality in Kernel app (all channels green)
   - Note: record fiducial positions (nasion, left/right preauricular) if available

2. **Start kernel-sdk** on acquisition laptop
   ```bash
   # Start acquisition + ZMQ relay to DGX
   python relay_to_dgx.py --dgx-ip 192.168.1.100 --port 15555
   ```

3. **Start dot-jax pipeline** on DGX Spark
   ```bash
   python examples/07_realtime_demo.py --live --port 15555
   # Open dashboard/index.html in browser
   ```

4. **Verify signal** — check dashboard shows clean HbO traces, GVTD bar green

### 3.2 Recording (10 min)

1. **Launch N-back task** on stimulus laptop
   ```bash
   python nback_dlpfc.py
   ```

2. **Monitor in real-time** on dashboard:
   - HbO timeseries should show clear block-locked oscillations
   - GVTD should stay low (no motion artifacts)
   - Event markers appear as vertical lines

3. **Subject instructions:**
   - "Sit comfortably, minimize head movement"
   - "You will see letters appear on screen"
   - "For 0-back: press spacebar when you see 'X'"
   - "For 2-back: press spacebar when the current letter matches the one from 2 letters ago"
   - "Between blocks, rest and look at the fixation cross"

### 3.3 Analysis (real-time + offline)

**Real-time (during recording):**
- dot-jax `RealtimePipeline` reconstructs HbO at each frame
- `EpochAccumulator` averages 2-back blocks to show emerging DLPFC activation
- Dashboard shows spatial activation map updating live

**Offline (after recording, ~2 min):**
```python
import jax; jax.config.update("jax_enable_x64", True)
from dot_jax.io import read_snirf, snirf_to_dot_jax
from dot_jax.hemodynamics import *
from dot_jax.realtime import RealtimePipeline
from dot_jax.atlas import generate_head_mesh, project_to_surface

# Load recorded data
snirf = read_snirf("session.snirf")
jd = snirf_to_dot_jax(snirf)

# Full preprocessing
raw = jd["data"]
od = intensity_to_od(raw)
artifacts = detect_motion_artifacts(od, fs=4.75)
od = correct_motion_spline(od, artifacts, fs=4.75)
od = bandpass_filter(od, fs=4.75, low=0.01, high=0.2)

# Block averaging: 2-back vs 0-back contrast
# ... extract epochs around block onsets ...
# ... compute t-statistic map across nodes ...

# Identify peak DLPFC activation
dlpfc_node = jnp.argmax(t_stat_map)
target_mni = mesh.node[dlpfc_node]  # MNI coordinate
print(f"DLPFC target: MNI ({target_mni[0]:.1f}, {target_mni[1]:.1f}, {target_mni[2]:.1f})")
```

---

## 4. Target Identification for SAINT

### 4.1 From HbO map to TMS target

1. **Compute 2-back > 0-back contrast** at each mesh node
   - Block-average HbO for 2-back epochs (5 blocks × 30s)
   - Block-average HbO for 0-back epochs (5 blocks × 30s)
   - t-statistic: `t = (mean_2back - mean_0back) / SE`

2. **Identify left DLPFC cluster**
   - Threshold t-map (e.g., t > 2.0)
   - Restrict to left frontal nodes (MNI x < 0, y > 0, z > 30)
   - Find peak activation coordinate

3. **Convert to scalp target**
   - The peak MNI coordinate maps to a scalp position
   - dot-jax's `project_to_surface` gives the nearest scalp point
   - Register to neuronavigation system or measure from 10-20 landmarks

### 4.2 Conventional SAINT target for reference

The standard fMRI-derived SAINT target is typically:
- **Left DLPFC:** MNI approximately (-42, 44, 26) to (-46, 38, 28)
- Individual variation: the sgACC-anticorrelated subregion varies by ~1–2 cm
- The fNIRS-derived target should fall within this general region

### 4.3 Validation approach

For the NeuroHack demo, we can validate by:
1. Running the N-back task with Kernel Flow
2. Processing with dot-jax to identify the DLPFC peak
3. Comparing the identified MNI coordinate to the canonical F3 position
4. If structural MRI is available: comparing to the fMRI-derived target

### 4.4 Spatial resolution considerations

| Modality | Spatial resolution | DLPFC localisation |
|----------|-------------------|-------------------|
| fMRI (SAINT standard) | ~2–3 mm | Subregion of DLPFC |
| HD-DOT (Kernel Flow + dot-jax) | ~10–15 mm | DLPFC region |
| Standard fNIRS | ~20–30 mm | General frontal area |
| 10-20 F3 rule-of-thumb | ~20 mm | Fixed anatomical target |

HD-DOT with Kernel Flow's 3000+ channels provides ~10–15 mm resolution —
substantially better than standard fNIRS and competitive with the precision
needed for TMS targeting (coil focal spot ~15–20 mm).

---

## 5. SAINT Stimulation Protocol (reference)

After target identification, the SAINT protocol delivers:

| Parameter | Value |
|-----------|-------|
| Stimulation type | Intermittent theta-burst (iTBS) |
| Pulses per session | 1800 (600 triplets at 50 Hz, every 200 ms) |
| Sessions per day | 10 (50 min inter-session interval) |
| Treatment days | 5 consecutive days |
| Total pulses | 90,000 |
| Intensity | 90% resting motor threshold |
| Target | Individualised left DLPFC (fMRI or fNIRS-guided) |

**Note:** The Kernel Flow headset should be **removed during TMS delivery**
(metallic SPAD detectors, laser sources). It can be reapplied between
sessions to monitor treatment response.

---

## 6. Safety Considerations

- **TMS safety:** Follow IFCN guidelines (Rossi et al., 2021). Screen for
  contraindications (epilepsy, metallic implants, cardiac devices).
- **Kernel Flow:** Non-invasive, low-power NIR lasers (Class 1). No known
  safety concerns. Remove during TMS.
- **fNIRS depth limitation:** sgACC is too deep (~40 mm from scalp) for
  fNIRS photon penetration (~15–20 mm). The SAINT protocol's specific
  innovation — targeting based on DLPFC-sgACC anticorrelation — cannot be
  directly replicated with fNIRS alone. Our approach uses a **different
  biomarker**: maximal DLPFC task activation during N-back, which is a proxy
  for the functionally relevant DLPFC subregion. The TRIBE structural
  connectivity prior (Section 7) partially compensates by predicting deep
  source (sgACC) involvement from white matter pathways that fNIRS cannot
  directly measure.
- **Ethics:** Standard institutional ethics approval for non-invasive brain
  stimulation research.

---

## 7. TRIBE-Predicted fNIRS Activation

### 7.1 Concept

The **TRIBE model** (Tractography-based Regional Individualised Brain
Estimation) predicts regional brain activity from structural connectivity.
Applied here, TRIBE could:

1. **Predict DLPFC activation from diffusion MRI** — use the structural
   connectome to estimate which DLPFC subregion would activate during the
   N-back task, before any fNIRS data is collected
2. **Serve as a Bayesian prior** — combine the TRIBE prediction with the
   fNIRS observation via SBI (simulation-based inference) to produce a
   posterior activation map that's better than either alone
3. **Bridge modalities** — TRIBE links structural MRI (which SAINT already
   requires for neuronavigation) to functional activation (which we measure
   with fNIRS), closing the loop

### 7.2 Integration with dot-jax

The TRIBE prediction produces an expected activation pattern over cortical
regions. In the dot-jax framework:

```python
# TRIBE predicts expected delta_mua at each mesh node
tribe_prior = predict_activation_from_connectome(connectome, task="nback")

# dot-jax forward model: what fNIRS signal would this produce?
expected_signal = forward_cw(mesh, mua_bg + tribe_prior, musp, srcpos, detpos)

# Compare to observed signal → update prior via SBI
posterior = sbi_update(prior=tribe_prior, observation=observed_signal,
                        forward_model=forward_cw)
```

Because dot-jax's forward model is differentiable, the TRIBE prior can be
refined by gradient descent to match the observed fNIRS data — effectively
fitting a structurally-informed functional activation map.

### 7.3 Advantages for SAINT targeting

- **Structural + functional constraint** — the target is informed by both
  white matter connectivity (TRIBE) and hemodynamic activation (fNIRS)
- **Handles fNIRS depth limitations** — TRIBE can predict deep source
  activity (sgACC) that fNIRS cannot directly measure
- **Pre-session prediction** — if structural MRI is available, TRIBE gives
  an initial target estimate before the fNIRS session even starts
- **Multi-modal SBI** — the JAX ecosystem (sbi4dwi for diffusion, dot-jax
  for fNIRS, vpjax for hemodynamics) enables joint inference across modalities

---

## 8. What's New (NeuroHack contribution)

1. **First fNIRS-guided SAINT target identification** — replacing the fMRI
   requirement with a 10-minute portable session
2. **Real-time DOT reconstruction on DGX Spark** — dot-jax processes Kernel
   Flow's 6990 channels through a differentiable forward model
3. **Live activation mapping** — the dashboard shows DLPFC activation emerging
   during the task, not after offline analysis
4. **TRIBE-informed prior** — structural connectivity predicts activation
   before fNIRS data, then fNIRS refines it via SBI
5. **Open source** — the entire pipeline (dot-jax + PsychoPy task + dashboard)
   is open, vs the proprietary Kernel + fMRI SAINT workflow
6. **Differentiable forward model** — dot-jax can optimize the head model
   to the individual's data via gradient descent, improving localisation
7. **Multi-modal SBI** — joint inference across structural MRI (TRIBE/sbi4dwi),
   fNIRS (dot-jax), and hemodynamics (vpjax) in one JAX computation graph

---

## References

1. Cole EJ, Stimpson KH, Bentzley BS, et al. "Stanford Accelerated Intelligent
   Neuromodulation Therapy for Treatment-Resistant Depression." Am J Psychiatry.
   2020;177(8):716-726.
2. Cole EJ, Phillips AL, Bentzley BS, et al. "Stanford Neuromodulation Therapy
   (SNT): A Double-Blind Randomized Controlled Trial." Am J Psychiatry.
   2022;179(2):132-141.
3. Fox MD, Buckner RL, White MP, et al. "Efficacy of transcranial magnetic
   stimulation targets for depression is related to intrinsic functional
   connectivity with the subgenual cingulate." Biol Psychiatry. 2012;72(7):595-603.
4. Hermes D, et al. "fNIRS-based DLPFC activation during N-back working memory
   task." NeuroImage. 2019.
5. Owen AM, McMillan KM, Laird AR, Bullmore E. "N-back working memory paradigm:
   a meta-analysis of normative functional neuroimaging studies." Hum Brain Mapp.
   2005;25(1):46-59.
6. Rossi S, Hallett M, Rossini PM, Pascual-Leone A. "Safety, ethical
   considerations, and application guidelines for the use of transcranial
   magnetic stimulation." Clin Neurophysiol. 2021.
