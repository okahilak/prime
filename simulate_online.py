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
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import torch

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


def main():
    if len(sys.argv) < 2:
        print("Usage: python simulate_online.py <subject_id>")
        sys.exit(1)
    subject_id = int(sys.argv[1])
    subject_id_str = f"sub-{subject_id:03d}"
    predictions_path = f"results/test/predictions_subj_{subject_id}.npz"

    print("=" * 70)
    print("SIMULATE ONLINE PROCESSING (trial-by-trial)")
    print(f"Subject: {subject_id_str}")
    print(f"Calibration trials: {N_CALIBRATION_TRIALS}")
    print("=" * 70)

    # --- Setup ---
    cfg = get_default_config()
    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"

    # --- Load all data (in a real system, trials would arrive one at a time) ---
    print("\nLoading raw data...")
    epochs = _load_subject_epochs(subject_id_str, cfg)
    all_eeg_data = epochs.get_data(copy=False)
    all_events = epochs.events
    n_total_trials = all_eeg_data.shape[0]
    print(f"Loaded {n_total_trials} total trials for {subject_id_str}")

    # =========================================================================
    # CALIBRATION PHASE — feed trials one at a time
    # =========================================================================
    print("\n" + "=" * 70)
    print("CALIBRATION PHASE")
    print("=" * 70)

    # --- Trial-by-trial accumulation ---
    calibrator = Calibrator(cfg, forward_path)
    dipole_fitter = DipoleFitter(forward_path)
    normalizer = TEPNormalizer(scale_factor=1.0)

    for trial_idx in range(N_CALIBRATION_TRIALS):
        # Simulate receiving a single trial
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)

        calibrator.add_trial(trial)

        if (trial_idx + 1) % 25 == 0:
            print(f"  Accumulated {trial_idx + 1}/{N_CALIBRATION_TRIALS} calibration trials")

    # --- We have enough trials: run calibration ---
    print("\nRunning calibration preprocessing...")
    cal_trials = calibrator.calibrate()
    print(f"Calibration preprocessing done, used {len(cal_trials)}/{N_CALIBRATION_TRIALS} trials.")

    # --- Calibrate dipole fitting parameters ---
    print("\nCalibrating dipole parameters...")
    dipole_fitter.fit(cal_trials)

    # --- Fit dipoles to calibration post-stim trials ---
    print("\nFitting dipoles to calibration trials...")
    cal_amplitudes = dipole_fitter.fit_trials(cal_trials, orientation=None)
    print(f"  {len(cal_amplitudes)} dipoles")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print(f"  Calibration params obtained: {list(calibrator.calibration_params.keys())}")
    print(f"  Calibration trials survived: {len(cal_trials)}")
    print("=" * 70)

    # =========================================================================
    # INTERVENTION PHASE — process post-calibration trials, fit dipoles,
    # compute normalized TEP labels using the same Calibrator preprocessing
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERVENTION PHASE — TEP label computation")
    print("=" * 70)

    n_intervention = n_total_trials - N_CALIBRATION_TRIALS
    print(f"Processing {n_intervention} intervention trials...")

    # --- Process intervention trials: require both pre & post to pass ---
    intervention_trials = []
    for trial_idx in range(N_CALIBRATION_TRIALS, n_total_trials):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        processed = calibrator.preprocess(trial)
        if processed is not None:
            intervention_trials.append(processed)
        if (trial_idx - N_CALIBRATION_TRIALS + 1) % 100 == 0:
            print(f"  Preprocessed {trial_idx - N_CALIBRATION_TRIALS + 1}/{n_intervention} "
                  f"intervention trials")

    print(f"  {len(intervention_trials)}/{n_intervention} intervention trials survived "
          f"(both pre & post)")

    print("\nFitting free-orientation dipoles to intervention trials...")
    int_amplitudes = dipole_fitter.fit_trials(intervention_trials, orientation=None)
    print(f"  {len(int_amplitudes)} dipoles fitted")

    # --- TEP normalization ---
    normalizer.fit(cal_amplitudes)
    all_labels = normalizer.transform(np.concatenate([cal_amplitudes, int_amplitudes]))

    # Extract intervention labels
    int_labels = all_labels[len(cal_amplitudes):]
    # Also compute calibration labels (needed for calibration fine-tuning)
    cal_labels = all_labels[:len(cal_amplitudes)]
    print(f"\n  Calibration TEP amplitudes: {len(cal_amplitudes)}")
    print(f"  Intervention TEP amplitudes: {len(int_amplitudes)}")
    print(f"  Intervention labels (first 5): {int_labels[:5]}")

    # =========================================================================
    # COMPARE WITH OFFLINE LABELS
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARE ONLINE vs OFFLINE LABELS")
    print("=" * 70)
    offline = np.load(predictions_path)
    offline_labels = offline["actual_values"]

    n_compare = min(len(int_labels), len(offline_labels))
    online_labels = int_labels[:n_compare]
    offline_labels_cmp = offline_labels[:n_compare]

    print(f"  Online labels count:  {len(int_labels)}")
    print(f"  Offline labels count: {len(offline_labels)}")
    print(f"  Comparing first {n_compare} labels...")
    print(f"")
    print(f"  First online label:   {online_labels[0]:.6f}")
    print(f"  First offline label:  {offline_labels_cmp[0]:.6f}")
    print(f"  Match (first):        {np.isclose(online_labels[0], offline_labels_cmp[0], atol=1e-4)}")
    print(f"")
    diffs = np.abs(online_labels - offline_labels_cmp)
    print(f"  Max absolute diff:    {np.max(diffs):.8f}")
    print(f"  Mean absolute diff:   {np.mean(diffs):.8f}")
    print(f"  Num diffs > 0.01:     {np.sum(diffs > 0.01)}")
    print(f"  Num diffs > 0.001:    {np.sum(diffs > 0.001)}")
    match_all = np.allclose(online_labels, offline_labels_cmp, atol=1e-7)
    print(f"  All match (atol=1e-7): {match_all}")
    print("=" * 70)

    # =========================================================================
    # CLASSIFIER — Load pretrained, calibrate, run online finetuning
    # =========================================================================
    print("\n" + "=" * 70)
    print("CLASSIFIER — Pretrained model + Calibration + Online finetuning")
    print("=" * 70)

    # Torch and NumPy setup
    device = torch.device("cuda")
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.cuda.manual_seed_all(CONFIG["seed"])

    # Deterministic settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    # --- Prepare pre-stim EEG data (matching offline tmin/tmax crop) ---
    cal_pre_epochs = mne.concatenate_epochs([t.epoch_pre for t in cal_trials])
    cal_pre_data = cal_pre_epochs.copy().crop(
        tmin=CONFIG["tmin"], tmax=CONFIG["tmax"], include_tmax=True
    ).get_data(copy=False)
    print(f"\n  Pre-stim EEG shape: ({cal_pre_data.shape[1]} ch, {cal_pre_data.shape[2]} timepoints)")
    print(f"  Calibration trials for classifier: {cal_pre_data.shape[0]}")

    # --- Create the OnlinePredictor (single instance for calibration + online) ---
    global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=False)
    predictor = OnlinePredictor(global_backrotation, model_state_dict=checkpoint["model_state_dict"])
    print(f"  Loaded pretrained model from: {PRETRAINED_MODEL_PATH}")

    # --- STAGE 2: CALIBRATION FINE-TUNING ---
    print("\n  --- Calibration (using OnlinePredictor.calibrate) ---")
    predictor.calibrate(cal_pre_data, cal_labels)
    print(f"  Calibration complete ({len(cal_pre_data)} trials, "
          f"{CONFIG['calibration_epochs']} epochs)")

    # --- STAGE 3: ONLINE FINE-TUNING (trial-by-trial) ---
    print("\n  --- Online fine-tuning simulation ---")

    # Prepare intervention pre-stim data
    int_pre_epochs = mne.concatenate_epochs([t.epoch_pre for t in intervention_trials])
    int_pre_data = int_pre_epochs.copy().crop(
        tmin=CONFIG["tmin"], tmax=CONFIG["tmax"], include_tmax=True
    ).get_data(copy=False)
    n_online_trials = len(int_pre_data)

    # Restrict to first 150 trials
    n_online_trials = min(n_online_trials, 150)

    print(f"  Online trials: {n_online_trials}")

    # Reset RNG state for deterministic behavior during online simulation
    predictor.prepare_for_stream(CONFIG["seed"])

    all_predictions = []

    for trial_idx in range(n_online_trials):
        single_epoch_np = int_pre_data[trial_idx]
        single_label_np = int_labels[trial_idx]

        # --- PREDICT ---
        pred_prob = predictor.predict(single_epoch_np)
        all_predictions.append(pred_prob)

        # --- FINETUNE (adapt alignment + buffered supervised update) ---
        predictor.finetune(single_epoch_np, single_label_np)

        if (trial_idx + 1) % 100 == 0:
            print(f"    Trial {trial_idx + 1}/{n_online_trials}")

    all_predictions = np.array(all_predictions)
    print(f"\n  Online simulation complete. Predictions: {len(all_predictions)}")
    print(f"  First 5 predictions: {all_predictions[:5]}")

    # =========================================================================
    # COMPARE WITH OFFLINE PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARE ONLINE vs OFFLINE")
    print("=" * 70)
    offline = np.load(predictions_path)
    offline_preds = offline["predictions"]
    offline_labels = offline["actual_values"]

    # --- Labels comparison ---
    n_compare = min(len(int_labels), len(offline_labels))
    label_diffs = np.abs(int_labels[:n_compare] - offline_labels[:n_compare])
    print(f"\n  LABELS ({n_compare} trials):")
    print(f"    Max diff:       {np.max(label_diffs):.2e}")
    print(f"    Num diffs > 0:  {np.sum(label_diffs > 0)}")
    print(f"    Exact match:    {np.array_equal(int_labels[:n_compare], offline_labels[:n_compare])}")

    # --- Predictions comparison ---
    n_compare_pred = min(len(all_predictions), len(offline_preds))
    pred_diffs = np.abs(all_predictions[:n_compare_pred] - offline_preds[:n_compare_pred])
    preds_all_close = np.allclose(
        all_predictions[:n_compare_pred], offline_preds[:n_compare_pred], atol=1e-4
    )
    print(f"\n  PREDICTIONS ({n_compare_pred} trials):")
    print(f"    Max diff:       {np.max(pred_diffs):.6f}")
    print(f"    Mean diff:      {np.mean(pred_diffs):.6f}")
    print(f"    Num exact:      {np.sum(pred_diffs == 0)}/{n_compare_pred}")
    print(f"    Num < 1e-6:     {np.sum(pred_diffs < 1e-6)}")
    print(f"    Num < 1e-4:     {np.sum(pred_diffs < 1e-4)}")
    print(f"    Num < 0.001:    {np.sum(pred_diffs < 0.001)}")
    print(f"    All close (1e-4): {preds_all_close}")
    print("=" * 70)

    labels_match = np.array_equal(int_labels[:n_compare], offline_labels[:n_compare])
    if not labels_match:
        print("\nRESULT: FAIL — Labels do not match.")
        sys.exit(1)

    if not preds_all_close:
        print("\nRESULT: FAIL — Predictions do not match within tolerance.")
        sys.exit(1)

    print("\nRESULT: PASS — all results match.")
    sys.exit(0)


if __name__ == "__main__":
    main()
