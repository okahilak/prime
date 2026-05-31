"""Simulate the C++ backend feeding buffers through prime.py's process_event.

This script reads the CSV+JSON simulator dataset, extracts buffers around each
event (mimicking what the C++ NeuroSimo backend does), and feeds them through
the Decider class in prime.py. Results are compared against simulate_online.py
run on the same data.

Usage (from workspace root, with venv activated):
    python simulation/simulate_via_prime.py [--n-trials N]

Examples:
    python simulation/simulate_via_prime.py --n-trials 5    # First 5 calib trials (data check)
    python simulation/simulate_via_prime.py --n-trials 130  # 125 calib + 5 intervention
"""

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np

# Add paths for imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "prime"))
sys.path.insert(0, str(ROOT_DIR / "prime" / "online_preprocessing"))

from prime import Decider, EVENT_SAMPLE_WINDOW
from online_preprocessing.calibrator import Calibrator
from online_preprocessing.dipole_fitter import DipoleFitter
from online_preprocessing.trial_loader_from_csv import TrialLoaderFromCsv
from tep_normalizer import TEPNormalizer
from online_predictor import OnlinePredictor

# Same constants as simulate_online.py
N_CALIBRATION_TRIALS = 125
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
SEED = 42


def load_dataset(json_path):
    """Load continuous data and event times from the simulator dataset."""
    json_path = Path(json_path)
    with open(json_path) as f:
        metadata = json.load(f)

    dataset_dir = json_path.parent
    sfreq = metadata["session"]["sampling_frequency"]
    n_eeg_channels = metadata["session"]["num_eeg_channels"]
    n_emg_channels = metadata["session"]["num_emg_channels"]

    event_times = np.loadtxt(dataset_dir / metadata["event_file"])

    print(f"Loading CSV data from {metadata['data_file']}...")
    raw_data = np.loadtxt(dataset_dir / metadata["data_file"], delimiter=',')
    print(f"  Data shape: {raw_data.shape}")

    return raw_data, event_times, sfreq, n_eeg_channels, n_emg_channels


def extract_buffer(raw_data, event_sample, sfreq, window, n_eeg_channels, n_emg_channels):
    """Extract EEG/EMG buffers and time_offsets around an event.

    Mimics what the C++ backend does: provides samples covering the window
    [window[0], window[1]] around the event at the given sampling frequency.

    The number of samples is: int(round((window[1] - window[0]) * sfreq)) + 1
    (same formula as TrialLoaderFromCsv uses for n_times).
    """
    start_offset = int(round(window[0] * sfreq))
    n_samples = int(round((window[1] - window[0]) * sfreq)) + 1

    start_sample = event_sample + start_offset
    end_sample = start_sample + n_samples

    eeg_buffer = raw_data[start_sample:end_sample, :n_eeg_channels]
    if n_emg_channels > 0:
        emg_buffer = raw_data[start_sample:end_sample, n_eeg_channels:n_eeg_channels + n_emg_channels]
    else:
        emg_buffer = np.zeros((n_samples, 0))

    # Time of each sample relative to the event
    time_offsets = np.arange(n_samples) / sfreq + window[0]

    return eeg_buffer, emg_buffer, time_offsets


def run_comparison(n_trials):
    """Run both reference and Decider paths for n_trials and compare."""

    dataset_path = DATA_ROOT / "simulator" / "sub-021" / "sub-021-short.json"
    ref_json_path = DATA_ROOT / "simulator" / "sub-021" / "sub-021-short.json"
    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
    pretrained_model_path = str(ROOT_DIR / "prime" / "results" / "train" / "pretrained.pt")
    global_backrotation_path = str(ROOT_DIR / "prime" / "results" / "train" / "global_backrotation.npy")

    # --- Load data ---
    raw_data, event_times, sfreq, n_eeg_channels, n_emg_channels = load_dataset(dataset_path)
    event_samples = np.round(event_times * sfreq).astype(int)
    n_trials = min(n_trials, len(event_times))
    print(f"\nRunning {n_trials} trials (calibration={N_CALIBRATION_TRIALS})")
    print(f"Event sample window: {EVENT_SAMPLE_WINDOW}")
    print(f"Samples per trial: {int(round((EVENT_SAMPLE_WINDOW[1] - EVENT_SAMPLE_WINDOW[0]) * sfreq)) + 1}")

    # --- Load reference trial loader ---
    trial_loader = TrialLoaderFromCsv(ref_json_path)

    # --- First: verify raw data matches between the two approaches ---
    print("\n--- Verifying raw trial data matches ---")
    for i in range(min(3, n_trials)):
        ref_trial = trial_loader.get_trial(i)
        ref_data = ref_trial.get_data(copy=False)[0]  # (n_channels, n_times)

        eeg_buffer, _, time_offsets = extract_buffer(
            raw_data, event_samples[i], sfreq,
            EVENT_SAMPLE_WINDOW, n_eeg_channels, n_emg_channels,
        )
        dec_data = eeg_buffer.T  # (n_channels, n_times)

        # Check shapes
        if ref_data.shape != dec_data.shape:
            print(f"  Trial {i}: SHAPE MISMATCH ref={ref_data.shape} dec={dec_data.shape}")
            continue

        max_diff = np.max(np.abs(ref_data - dec_data))
        print(f"  Trial {i}: shape={ref_data.shape}  max_diff={max_diff:.2e}  "
              f"tmin_ref={ref_trial.tmin:.4f}  tmin_dec={time_offsets[0]:.4f}")

    # --- Run reference pipeline ---
    print("\n--- Running REFERENCE (simulate_online.py logic) ---")
    global_backrotation = np.load(global_backrotation_path)
    ref_predictor = OnlinePredictor(global_backrotation, model_path=pretrained_model_path, seed=SEED)
    ref_calibrator = Calibrator(forward_path)
    ref_dipole_fitter = DipoleFitter(forward_path)
    ref_normalizer = TEPNormalizer()

    for trial_idx in range(min(N_CALIBRATION_TRIALS, n_trials)):
        ref_calibrator.add_raw_trial(trial_loader.get_trial(trial_idx))

    ref_labels = []
    ref_predictions = []
    if n_trials > N_CALIBRATION_TRIALS:
        trials = ref_calibrator.calibrate()
        amplitudes = ref_dipole_fitter.calibrate(trials)
        labels = ref_normalizer.calibrate(amplitudes)
        ref_predictor.calibrate(trials, labels)

        for trial_idx in range(N_CALIBRATION_TRIALS, n_trials):
            trial = ref_calibrator.preprocess(trial_loader.get_trial(trial_idx))
            if trial is None:
                print(f"  Ref trial {trial_idx}: REJECTED")
                continue
            amplitude = ref_dipole_fitter.fit_trial(trial)
            label = ref_normalizer.transform(amplitude)
            probability = ref_predictor.predict(trial)
            ref_predictor.finetune(trial, label)
            ref_labels.append(label)
            ref_predictions.append(probability)
            print(f"  Ref trial {trial_idx}: pred={probability:.6f}  label={label:.6f}")

    # --- Run Decider pipeline ---
    print("\n--- Running DECIDER (prime.py process_event) ---")
    decider = Decider(
        subject_id="sub-021-short",
        num_eeg_channels=n_eeg_channels,
        num_emg_channels=n_emg_channels,
        sampling_frequency=int(sfreq),
    )

    dec_labels = []
    dec_predictions = []

    for trial_idx in range(n_trials):
        eeg_buffer, emg_buffer, time_offsets = extract_buffer(
            raw_data, event_samples[trial_idx], sfreq,
            EVENT_SAMPLE_WINDOW, n_eeg_channels, n_emg_channels,
        )

        result = decider.process_event(
            reference_time=event_times[trial_idx],
            reference_index=int(event_samples[trial_idx]),
            time_offsets=time_offsets,
            eeg_buffer=eeg_buffer,
            emg_buffer=emg_buffer,
            is_coil_at_target=True,
            stage_name="intervention",
            trial_in_stage=trial_idx,
        )

        if result is not None:
            dec_predictions.append(result["prediction"])
            dec_labels.append(result["label"])
            print(f"  Dec trial {trial_idx}: pred={result['prediction']:.6f}  label={result['label']:.6f}")

    # --- Compare ---
    ref_labels = np.array(ref_labels)
    ref_predictions = np.array(ref_predictions)
    dec_labels = np.array(dec_labels)
    dec_predictions = np.array(dec_predictions)

    print("\n" + "=" * 70)
    print("COMPARISON: Reference vs Decider")
    print("=" * 70)

    if n_trials <= N_CALIBRATION_TRIALS:
        print(f"\n  Only calibration trials run ({n_trials}/{N_CALIBRATION_TRIALS}).")
        print("  Raw data verified above. No predictions to compare yet.")
        print("=" * 70)
        return

    if len(ref_predictions) != len(dec_predictions):
        print(f"\n  COUNT MISMATCH: ref={len(ref_predictions)}, decider={len(dec_predictions)}")
        n_compare = min(len(ref_predictions), len(dec_predictions))
    else:
        n_compare = len(ref_predictions)
        print(f"\n  Trial counts match: {n_compare}")

    if n_compare > 0:
        label_diffs = np.abs(ref_labels[:n_compare] - dec_labels[:n_compare])
        pred_diffs = np.abs(ref_predictions[:n_compare] - dec_predictions[:n_compare])

        labels_match = np.allclose(ref_labels[:n_compare], dec_labels[:n_compare], atol=1e-7)
        preds_match = np.allclose(ref_predictions[:n_compare], dec_predictions[:n_compare], atol=1e-4)

        print(f"\n  LABELS ({n_compare} trials):")
        print(f"    Max diff:         {np.max(label_diffs):.2e}")
        print(f"    Mean diff:        {np.mean(label_diffs):.2e}")
        print(f"    All close (1e-7): {labels_match}")

        print(f"\n  PREDICTIONS ({n_compare} trials):")
        print(f"    Max diff:         {np.max(pred_diffs):.6f}")
        print(f"    Mean diff:        {np.mean(pred_diffs):.6f}")
        print(f"    All close (1e-4): {preds_match}")

        # Print all trial results for easy inspection
        print(f"\n  Per-trial results:")
        for i in range(n_compare):
            match_str = "OK" if (label_diffs[i] < 1e-7 and pred_diffs[i] < 1e-4) else "DIFF"
            print(f"    [{match_str}] Trial {N_CALIBRATION_TRIALS + i}: "
                  f"ref_pred={ref_predictions[i]:.6f} dec_pred={dec_predictions[i]:.6f} "
                  f"ref_label={ref_labels[i]:.6f} dec_label={dec_labels[i]:.6f}")

        print("=" * 70)
        if labels_match and preds_match:
            print("\nRESULT: PASS")
        else:
            print("\nRESULT: FAIL")
    else:
        print("\n  No intervention trials to compare.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare Decider vs simulate_online.py")
    parser.add_argument("--n-trials", type=int, default=130,
                        help="Total trials to process (default: 130 = 125 calib + 5 intervention)")
    args = parser.parse_args()

    run_comparison(args.n_trials)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
