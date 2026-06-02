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

# Force single-threaded BLAS/LAPACK BEFORE importing numpy/scipy/torch.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import mne
import numpy as np

# --- Local imports ---
from prime.prime_config import get_raw_post_time_range, get_raw_pre_time_range
from prime.online_preprocessing.preprocessor import Preprocessor, crop_mne_trial_to_raw_epochs
from prime.online_preprocessing.dipole_fitter import DipoleFitter
from prime.tep_normalizer import TEPNormalizer
from prime.online_predictor import OnlinePredictor
from prime.online_preprocessing.trial_loader import TrialLoader
from prime.online_preprocessing.trial_loader_from_csv import TrialLoaderFromCsv

# =============================================================================
# Hard-coded constants
# =============================================================================
N_CALIBRATION_TRIALS = 125

PRETRAINED_MODEL_PATH = Path("prime") / "results" / "train" / "pretrained.pt"
GLOBAL_BACKROTATION_PATH = Path("prime") / "results" / "train" / "global_backrotation.npy"

SEED = 42


@contextmanager
def profile(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[profile] {label}: {elapsed * 1000:.1f}ms")


def print_summary(summary_text):
    print("\n" + "=" * 70)
    print(summary_text)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Simulate online processing for a single subject.")
    parser.add_argument("subject_id", type=int, help="Subject ID (e.g. 21 for sub-021)")
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Load trials from CSV simulator dataset instead of raw epochs",
    )
    args = parser.parse_args()

    subject_id = args.subject_id
    subject_id_str = f"sub-{subject_id:03d}"
    predictions_path = Path("prime") / "results" / "test" / f"predictions_subj_{subject_id}.npz"

    print(f"Subject: {subject_id_str}")
    print(f"Calibration trials: {N_CALIBRATION_TRIALS}")

    # --- Setup ---

    forward_path = Path("data") / "fsaverage" / "fsaverage-fwd.fif"

    # --- Load all data (in a real system, trials would arrive one at a time) ---
    print("\nLoading raw data...")

    if args.csv:
        json_path = Path("data") / "simulator" / subject_id_str / f"{subject_id_str}.json"
        trial_loader = TrialLoaderFromCsv(json_path)
    else:
        trial_loader = TrialLoader(subject_id_str)

    n_total_trials = trial_loader.num_trials
    print(f"Loaded {n_total_trials} trials for {subject_id_str}")

    # Calibration phase
    print_summary("CALIBRATION PHASE")

    global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)

    predictor = OnlinePredictor(global_backrotation, model_path=PRETRAINED_MODEL_PATH, seed=SEED)

    raw_pre_tmin, raw_pre_tmax = get_raw_pre_time_range()
    raw_post_tmin, raw_post_tmax = get_raw_post_time_range()
    preprocessor = Preprocessor(forward_path)
    dipole_fitter = DipoleFitter(forward_path)
    normalizer = TEPNormalizer()

    for trial_idx in range(N_CALIBRATION_TRIALS):
        raw_pre, raw_post = crop_mne_trial_to_raw_epochs(
            trial_loader.get_trial(trial_idx),
            raw_pre_tmin, raw_pre_tmax, raw_post_tmin, raw_post_tmax,
        )
        with profile("add_raw_pre"):
            preprocessor.add_raw_pre(raw_pre)
        with profile("add_raw_post"):
            preprocessor.add_raw_post(raw_post)

    cal_pre, cal_post = preprocessor.calibrate()
    amplitudes = dipole_fitter.calibrate(cal_post)
    labels = normalizer.calibrate(amplitudes)
    predictor.calibrate(cal_pre, labels)

    # Intervention phase
    print_summary("INTERVENTION PHASE")

    intervention_labels = []
    predictions = []
    for trial_idx in range(N_CALIBRATION_TRIALS, n_total_trials):
        if trial_idx % 100 == 0:
            print(f"Processing trial {trial_idx + 1}/{n_total_trials}...")

        raw_pre, raw_post = crop_mne_trial_to_raw_epochs(
            trial_loader.get_trial(trial_idx),
            raw_pre_tmin, raw_pre_tmax, raw_post_tmin, raw_post_tmax,
        )
        with profile("preprocess_pre"):
            processed_pre = preprocessor.preprocess_pre(raw_pre)
        with profile("preprocess_post"):
            processed_post = preprocessor.preprocess_post(raw_post)

        if processed_pre is None or processed_post is None:
            print(f"Trial {trial_idx + 1}: REJECTED by preprocessing")
            continue

        with profile("predict"):
            probability = predictor.predict(processed_pre)

        with profile("finetune"):
            amplitude = dipole_fitter.fit_trial(processed_post)
            label = normalizer.transform(amplitude)
            predictor.finetune(processed_pre, label)

        intervention_labels.append(label)
        predictions.append(probability)

        print(f"Trial {trial_idx + 1}: prediction={probability:.6f}  label={label:.6f}")

    intervention_labels = np.array(intervention_labels)
    predictions = np.array(predictions)

    # Compare with offline results
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
    predictions_all_close = np.allclose(predictions, offline_predictions, atol=1e-4)

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
    print(f"    All close (1e-4):   {predictions_all_close}")
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
