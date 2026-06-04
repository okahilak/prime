# PRIME-TEP Validation

## Experimental Design Considerations

### Overview

- 15 subjects, 2 sessions each (TBS, 3 pulses at 100 Hz), session order randomized:
  - PRIME high excitability
  - Brain-state-independent stimulation (PRIME only used for prediction performance evaluation, not to guide stimulation)

### Setup

- 62 EEG channels (-> REFTEP++)
- 1 EMG channel (FDI)
- 1 ECG channel
- 2 EOG channels
- cool-B65 coil (-> REFTEP++)
- Noise masking

### Stimulation Target and Intensity

- Left M1 hotspot with 110% RMT
- Adjustable based on TEP fidelity
- Aim: 10 µV peak-to-peak (30–60 ms post-TMS, visual inspection)
- Approx. 10–20 trials (RT-TEP/NeurOne)

## Experimental Design

### 1. Calibration Phase

- 100 valid brain-state-independent trials (max. 125 attempts)
- ITI: 6.5 ± 2.5 s
- Goal: use trials to calculate
  - Filters (SOUND, SSP-SIR, ICA)
  - Trial rejection thresholds
  - TEP estimates
- Offline subject-specific calibration of PRIME using preprocessed calibration data
- Use pretrained model (REFTEP++ data)

### 2. Intervention Phase

**Block design:**

- 4 blocks × 200 trials
- In PRIME session:
  - Randomized PRIME-guided and brain-state-independent trials (75/25)
- In brain-state-independent session:
  - 100% brain-state-independent trials

**Rolling window preprocessing:**

- Start: 2.5 s after pulse
- Step size: 10 ms (TODO: test system compatibility)
- Window size: 200 ms
- No stimulation if last 1 second of EEG data are rejected (TODO: not implemented; do not implement before clarifying)

### 3. Evaluation Phase

- Goal: assess lasting effects of the intervention
- 100 single TMS pulses each, 3 ± 0.5 s intervals, 110% RMT, at 0-, 15-, 30-, and 60-min post-intervention

## Follow-Up Experiments

- Test PRIME on Timo's data (prefrontal target)
- If successful -> closed-loop experiment

## Responsible Persons

- PRIME integration into NeuroSimo: Olli-Pekka
- PRIME experiment: Dania
