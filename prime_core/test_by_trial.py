"""
Simulate real-time (online) processing for a single subject, trial by trial.

This script processes data sequentially, one trial at a time, to mirror
what a true online system would do. Currently implements the calibration phase:

1. Feed trials one at a time, accumulating pre/post/ica epochs.
2. Once N_CALIBRATION_TRIALS are accumulated, run batch calibration preprocessing.
3. After calibration: calibrate dipole parameters from the averaged post-stim data.
4. Preprocess each calibration trial's pre and post periods using calibration params.
5. Fit dipoles to the calibration post-stim epochs.

Hard-coded settings:
- Subject: provided via command-line argument (integer, e.g. 21)
- Number of calibration trials: 125
"""

import os
import sys


def _configure_threading_from_cli() -> bool:
    # Must run before importing numeric libraries so BLAS/OpenMP picks it up.
    multi_threaded = "--multi-threaded" in sys.argv
    if not multi_threaded:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
    return not multi_threaded


SINGLE_THREADED = _configure_threading_from_cli()

import argparse
import time
from pathlib import Path

import numpy as np

# --- Local imports ---
from prime_core.prime_config import get_calibration_time_range
from prime_core.preprocessing.preprocessor import Preprocessor, crop_mne_trial_to_buffer
from prime_core.preprocessing.dipole_fitter import DipoleFitter
from prime_core.tep_normalizer import TEPNormalizer
from prime_core.online_predictor import OnlinePredictor
from prime_core.preprocessing.trial_loader import TrialLoader
from prime_core.preprocessing.trial_loader_from_csv import TrialLoaderFromCsv

# =============================================================================
# Hard-coded constants
# =============================================================================
N_CALIBRATION_TRIALS = 125

PRETRAINED_MODEL_PATH = Path("offline_results") / "train" / "pretrained.pt"
GLOBAL_BACKROTATION_PATH = Path("offline_results") / "train" / "global_backrotation.npy"

SEED = 42


def print_summary(summary_text):
    print("\n" + "=" * 70)
    print(summary_text)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Simulate online processing for a single subject.")
    parser.add_argument("subject_id", type=int, help="Subject ID (e.g. 21 for sub-021)")
    parser.add_argument(
        "--multi-threaded",
        action="store_true",
        help="Allow multi-threaded BLAS/OpenMP execution (default: single-threaded)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Load trials from CSV simulator dataset instead of raw epochs",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Total trials to process (default: all loaded trials)",
    )
    args = parser.parse_args()

    subject_id = args.subject_id
    subject_id_str = f"sub-{subject_id:03d}"
    predictions_path = Path("offline_results") / "test" / f"predictions_subj_{subject_id}.npz"

    print(f"Subject: {subject_id_str}")
    print(f"Calibration trials: {N_CALIBRATION_TRIALS}")
    print(f"Single-threaded: {SINGLE_THREADED}")

    # --- Setup ---

    forward_path = Path("offline_data") / "fsaverage" / "fsaverage-fwd.fif"

    # --- Load all data (in a real system, trials would arrive one at a time) ---
    print("\nLoading raw data...")

    if args.csv:
        json_path = Path("offline_data") / "simulator" / subject_id_str / f"{subject_id_str}.json"
        trial_loader = TrialLoaderFromCsv(json_path)
    else:
        trial_loader = TrialLoader(subject_id_str)

    n_total_trials = trial_loader.num_trials
    n_trials = n_total_trials if args.n_trials is None else min(args.n_trials, n_total_trials)
    print(f"Loaded {n_total_trials} trials for {subject_id_str}")
    print(f"Running {n_trials} trials")

    # Calibration phase
    print_summary("CALIBRATION PHASE")

    global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)

    predictor = OnlinePredictor(global_backrotation, model_path=PRETRAINED_MODEL_PATH, seed=SEED)

    trial_tmin, trial_tmax = get_calibration_time_range()
    preprocessor = Preprocessor(forward_path)
    dipole_fitter = DipoleFitter(forward_path)
    normalizer = TEPNormalizer()

    for trial_idx in range(N_CALIBRATION_TRIALS):
        eeg_buffer, relative_timestamps = crop_mne_trial_to_buffer(
            trial_loader.get_trial(trial_idx),
            trial_tmin, trial_tmax,
        )
        preprocessor.add_trial(eeg_buffer, relative_timestamps)

    model_buffers, dipole_buffers = preprocessor.calibrate()
    amplitudes = dipole_fitter.calibrate(dipole_buffers)
    labels = normalizer.calibrate(amplitudes)
    predictor.calibrate(model_buffers, labels)

    # Intervention phase
    print_summary("INTERVENTION PHASE")

    intervention_labels = []
    predictions = []
    preprocess_pre_times: list[float] = []
    predict_times: list[float] = []
    for trial_idx in range(N_CALIBRATION_TRIALS, n_trials):
        if trial_idx % 100 == 0:
            print(f"Processing trial {trial_idx + 1}/{n_trials}...")

        eeg_buffer, relative_timestamps = crop_mne_trial_to_buffer(
            trial_loader.get_trial(trial_idx),
            trial_tmin, trial_tmax,
        )
        t0 = time.perf_counter()
        processed_pre = preprocessor.preprocess_pre(eeg_buffer, relative_timestamps)
        preprocess_pre_times.append(time.perf_counter() - t0)

        processed_post = preprocessor.preprocess_post(eeg_buffer, relative_timestamps)

        if processed_pre is None or processed_post is None:
            print(f"Trial {trial_idx + 1}: REJECTED by preprocessing")
            continue

        t0 = time.perf_counter()
        probability = predictor.predict(processed_pre)
        predict_times.append(time.perf_counter() - t0)

        amplitude = dipole_fitter.fit_trial(processed_post)
        label = normalizer.transform(amplitude)
        predictor.finetune(processed_pre, label)

        intervention_labels.append(label)
        predictions.append(probability)

        print(f"Trial {trial_idx + 1}: prediction={probability:.6f}  label={label:.6f}")

    for label, times_list in [("preprocess_pre", preprocess_pre_times), ("predict", predict_times)]:
        times = np.array(times_list)
        mean_s = np.mean(times)
        std_s = np.std(times, ddof=1) if len(times) > 1 else 0.0
        sem_s = std_s / np.sqrt(len(times)) if len(times) > 1 else 0.0
        print(f"{label}: mean={mean_s * 1000:.1f}ms, std={std_s * 1000:.1f}ms, SEM={sem_s * 1000:.1f}ms (n={len(times)})")

    if n_trials < n_total_trials:
        print("Not all trials were processed. Exiting.")
        return

    intervention_labels = np.array(intervention_labels)
    predictions = np.array(predictions)

    offline_data = np.load(predictions_path)
    offline_labels = offline_data["actual_values"]
    offline_predictions = offline_data["predictions"]

    if len(intervention_labels) != len(offline_labels):
        raise RuntimeError(
            f"Label count mismatch: online={len(intervention_labels)}, offline={len(offline_labels)}"
        )
    if len(predictions) != len(offline_predictions):
        raise RuntimeError(
            f"Prediction count mismatch: online={len(predictions)}, offline={len(offline_predictions)}"
        )

    label_differences = np.abs(intervention_labels - offline_labels)
    prediction_differences = np.abs(predictions - offline_predictions)

    labels_match = np.allclose(intervention_labels, offline_labels, atol=1e-7)
    predictions_all_close = np.allclose(predictions, offline_predictions, atol=1e-2)

    print("\n" + "=" * 70)
    print("COMPARE ONLINE vs OFFLINE")
    print("=" * 70)
    print(f"\n  LABELS ({len(intervention_labels)} trials):")
    print(f"    Max diff:           {np.max(label_differences):.2e}")
    print(f"    Mean diff:          {np.mean(label_differences):.2e}")
    print(f"    All close (1e-7):   {labels_match}")
    print(f"\n  PREDICTIONS ({len(predictions)} trials):")
    print(f"    Max diff:           {np.max(prediction_differences):.6f}")
    print(f"    Mean diff:          {np.mean(prediction_differences):.6f}")
    print(f"    All close (1e-2):   {predictions_all_close}")
    print("=" * 70)

    if not labels_match:
        print("\nRESULT: FAIL — Labels do not match.")
        sys.exit(1)

    if not predictions_all_close:
        print("\nRESULT: FAIL — Predictions do not match within tolerance.")
        sys.exit(1)

    print("\nRESULT: PASS — all results match.")
    sys.exit(0)


if __name__ == "__main__":
    main()
