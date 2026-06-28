# PRIME-TEP validation

## Experimental design considerations

---

# Overview

- PRIME has been validated to deliver single-pulse TMS at high cortical excitability states (TEP-N45 Amplitude). This study represents the first test of whether the results translate to theta-burst stimulation-induced plasticity.

- Key Question:

Can stimulation delivered during PRIME-defined high excitability cortical states enhance plasticity outcomes (TEP-N45 Amplitude) compared with stimulation delivered at random?

---

# Overview

- 15 subjects, 2 sessions each (TBS, 3 pulses at 100 Hz), session order randomized
  - PRIME high excitability
  - brain-state-independent (PRIME only used for prediction performance evaluation, not to guide stimulation)

## Setup

- 62 EEG channels (-> REFTEP++)
- 1 EMG channel (FDI)
- 1 ECG channel
- 2 EOG channels
- cool-B65 coil (-> REFTEP++)
- Noise masking

---

# Experimental design

## 0. TEP-thresholding:

Stimulation intensity with 110% RMT adjustable based on TEP fidelity:

- approx. 10–20 trials (RT-TEP/NeurOne)
- aim = 10 µV peak-to-peak TEP Amplitude (N45)(30–60 ms post-TMS, visual inspection)

## 1. Baseline phase:

100 single TMS pulses on left M1 hotspot with 110% RMT with 3±0.5 s intervals

---

# Experimental design

## 2. Calibration phase

- 125 brain-state-independent trials: single-pulse TMS
- ITI 6.5 +- 2.5s
- goal = get 100 valid trials to calculate
  - filters (SOUND, SSP-SIR, ICA)
  - trial rejection thresholds
  - TEP estimates
- offline subject-specific calibration of PRIME using preprocessed calibration data
  - use pretrained model (REFTEP++ data)

---

# Experimental design

## 3. Intervention phase

### Block design

- 4 blocks × 200 trials
- Stimulation intensity:
  - triplets: 80% RMT
  - single-pulse TMS: 110% RMT
- ITI: 6.5 ± 2.5 s
  - If PRIME does not trigger before the end of the ITI, a pulse is delivered at the end of the ITI, but the trial is excluded from PRIME-triggered analyses

- Each block in the PRIME session:
  - 150 PRIME-triggered and 50 brain-state-independent trials are interleaved
  - 150 triplets and 50 single-pulse TMS trials are interleaved, with 5 single-pulse trials per 20-trial mini-block
  - Triggering mode (PRIME vs. brain-state-independent) and stimulation type (triplet vs. single-pulse) are independent factors

- Each block in the brain-state-independent session:
  - 200 brain-state-independent trials
  - same triplet/single-pulse structure as in the PRIME session

- Rolling-window preprocessing:
  - Start: 2.5 s after pulse
  - Step size: 10 ms (TO DO: test system compatibility)
  - Window size: 500 ms

- No stimulation if the last 1 s of EEG data is rejected
---

# Experimental design

## 4. Post-Intervention (Evaluation) phase

- Goal: assess lasting effects of the stimulation
- 100 single TMS pulses each, 3±0.5 s intervals, 110% RMT, at 0-, 15-, 30-, and 60-min post-intervention
