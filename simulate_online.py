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

import sys
import time
import warnings
from pathlib import Path

import mne
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")

# --- Local imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "online_preprocessing"))

from online_preprocessing.config import get_default_config
from online_preprocessing.calibrator import Calibrator
from online_preprocessing.dipole_fitter import DipoleFitter
from online_preprocessing.preprocess import (
    _load_subject_epochs,
    _single_trial_epochs_from_arrays,
)
from tep_normalizer import TEPNormalizer
from online_predictor import OnlinePredictor

# =============================================================================
# Hard-coded constants
# =============================================================================
N_CALIBRATION_TRIALS = 125
DATA_ROOT = Path("~/prime-data").expanduser()

PRETRAINED_MODEL_PATH = "results/train/pretrained.pt"
GLOBAL_BACKROTATION_PATH = "results/train/global_backrotation.npy"

# Config matching configs/prime.yaml
CONFIG = {
    "tmin": -0.060,
    "tmax": -0.010,
    "use_tta": True,
    "alignment_type": "euclidean",
    "use_backrotation": True,
    "alignment_ref_ema_beta": 0.99,
    "alignment_cov_epsilon": 1e-6,
    "alignment_transform_epsilon": 1e-7,
    "tta_cov_buffer_size": 50,
    "finetune_mode": "full",
    "finetune_epochs": 1,
    "batch_size_finetune": 50,
    "lr_finetune": 0.0001,
    "weight_decay_finetune": 0.0,
    "lr_calibration": 0.0001,
    "calibration_epochs": 50,
    "window_size": 100,
    "seed": 42,
}


def print_summary(summary_text):
    print("\n" + "=" * 70)
    print(summary_text)
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python simulate_online.py <subject_id>")
        sys.exit(1)
    subject_id = int(sys.argv[1])
    subject_id_str = f"sub-{subject_id:03d}"
    predictions_path = f"results/test/predictions_subj_{subject_id}.npz"

    print(f"Subject: {subject_id_str}")
    print(f"Calibration trials: {N_CALIBRATION_TRIALS}")

    # --- Setup ---

    # Get default config
    cfg = get_default_config()
    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"

    # --- Load all data (in a real system, trials would arrive one at a time) ---
    print("\nLoading raw data...")
    epochs = _load_subject_epochs(subject_id_str, cfg)
    all_eeg_data = epochs.get_data(copy=False)
    all_events = epochs.events
    n_total_trials = all_eeg_data.shape[0]
    print(f"Loaded {n_total_trials} total trials for {subject_id_str}")

    # Calibration phase
    print_summary("CALIBRATION PHASE")

    global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)

    predictor = OnlinePredictor(global_backrotation, model_path=PRETRAINED_MODEL_PATH, seed=CONFIG["seed"])

    calibrator = Calibrator(cfg, forward_path)
    dipole_fitter = DipoleFitter(forward_path)
    normalizer = TEPNormalizer(scale_factor=1.0)

    for trial_idx in range(N_CALIBRATION_TRIALS):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        calibrator.add_trial(trial)

    calibration_trials = calibrator.calibrate()
    calibration_amplitudes = dipole_fitter.calibrate(calibration_trials)
    calibration_labels = normalizer.calibrate(calibration_amplitudes)
    predictor.calibrate(calibration_trials, calibration_labels)

    # Intervention phase
    print_summary("INTERVENTION PHASE")

    intervention_labels = []
    online_predictions = []
    for trial_idx in range(N_CALIBRATION_TRIALS, n_total_trials):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        processed = calibrator.preprocess(trial)

        if processed is None:
            continue

        amplitude = dipole_fitter.fit_trial(processed, orientation=None)
        label = normalizer.transform(amplitude)
        probability = predictor.predict(processed)
        predictor.finetune(processed, label)

        intervention_labels.append(label)
        online_predictions.append(probability)

    intervention_labels = np.array(intervention_labels)
    online_predictions = np.array(online_predictions)

    # Compare with offline results
    offline_data = np.load(predictions_path)
    offline_labels = offline_data["actual_values"]
    offline_predictions = offline_data["predictions"]

    if len(intervention_labels) != len(offline_labels):
        raise RuntimeError(
            f"Label count mismatch: online={len(intervention_labels)}, offline={len(offline_labels)}"
        )
    if len(online_predictions) != len(offline_predictions):
        raise RuntimeError(
            f"Prediction count mismatch: online={len(online_predictions)}, offline={len(offline_predictions)}"
        )

    label_differences = np.abs(intervention_labels - offline_labels)
    prediction_differences = np.abs(online_predictions - offline_predictions)

    labels_match = np.allclose(intervention_labels, offline_labels, atol=1e-7)
    predictions_all_close = np.allclose(online_predictions, offline_predictions, atol=1e-4)

    print("\n" + "=" * 70)
    print("COMPARE ONLINE vs OFFLINE")
    print("=" * 70)
    print(f"\n  LABELS ({len(intervention_labels)} trials):")
    print(f"    Max diff:           {np.max(label_differences):.2e}")
    print(f"    Mean diff:          {np.mean(label_differences):.2e}")
    print(f"    All close (1e-7):   {labels_match}")
    print(f"\n  PREDICTIONS ({len(online_predictions)} trials):")
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
