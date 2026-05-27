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
- Subject: sub-018
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

# =============================================================================
# Hard-coded constants
# =============================================================================
SUBJECT_ID = "sub-018"
N_CALIBRATION_TRIALS = 125
DATA_ROOT = Path("~/prime-data").expanduser()


def main():
    print("=" * 70)
    print("SIMULATE ONLINE PROCESSING (trial-by-trial)")
    print(f"Subject: {SUBJECT_ID}")
    print(f"Calibration trials: {N_CALIBRATION_TRIALS}")
    print("=" * 70)

    # --- Setup ---
    cfg = get_default_config()
    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"

    # --- Load all data (in a real system, trials would arrive one at a time) ---
    print("\nLoading raw data...")
    epochs = _load_subject_epochs(SUBJECT_ID, cfg)
    all_eeg_data = epochs.get_data(copy=False)
    all_events = epochs.events
    n_total_trials = all_eeg_data.shape[0]
    print(f"Loaded {n_total_trials} total trials for {SUBJECT_ID}")

    # =========================================================================
    # CALIBRATION PHASE — feed trials one at a time
    # =========================================================================
    print("\n" + "=" * 70)
    print("CALIBRATION PHASE")
    print("=" * 70)

    # --- Trial-by-trial accumulation ---
    calibrator = Calibrator(cfg, forward_path)

    for trial_idx in range(N_CALIBRATION_TRIALS):
        # Simulate receiving a single trial
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)

        calibrator.add_trial(trial)

        if (trial_idx + 1) % 25 == 0:
            print(f"  Accumulated {trial_idx + 1}/{N_CALIBRATION_TRIALS} calibration trials")

    # --- We have enough trials: run calibration ---
    print("\nRunning calibration preprocessing...")
    t0 = time.time()
    n_successful_trials = calibrator.calibrate()
    print(f"Calibration preprocessing done in {time.time() - t0:.2f}s, "
          f"used {n_successful_trials} trials")

    # --- Calibrate dipole from calibration post-stim data ---
    # First, preprocess each calibration trial's post-stim to get clean epochs for dipole calibration
    print("\nPreprocessing calibration trials (post-stim) for dipole calibration...")
    cal_post_list = []
    for trial_idx in range(N_CALIBRATION_TRIALS):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        result_post = calibrator.preprocess_post(trial)
        if result_post is not False:
            cal_post_list.append(result_post)

    print(f"  {len(cal_post_list)}/{N_CALIBRATION_TRIALS} post-stim trials survived preprocessing")
    cal_post_epochs = mne.concatenate_epochs(cal_post_list)

    # --- Calibrate dipole fitting parameters ---
    print("\nCalibrating dipole parameters...")
    dipole_fitter = DipoleFitter(forward_path)
    dipole_fitter.fit(cal_post_epochs)
    position_index = dipole_fitter.fitting_info['position_index']
    tmin_fit, tmax_fit = dipole_fitter.fitting_info['time_range']
    print(f"  Position index: {position_index}")
    print(f"  Time range: [{tmin_fit*1000:.1f}, {tmax_fit*1000:.1f}] ms")

    # --- Preprocess calibration trials (pre-stim) for classifier input ---
    print("\nPreprocessing calibration trials (pre-stim) for classifier...")
    cal_pre_list = []
    for trial_idx in range(N_CALIBRATION_TRIALS):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        result_pre = calibrator.preprocess_pre(trial)
        if result_pre is not False:
            cal_pre_list.append(result_pre)

    print(f"  {len(cal_pre_list)}/{N_CALIBRATION_TRIALS} pre-stim trials survived preprocessing")
    cal_pre_epochs = mne.concatenate_epochs(cal_pre_list)

    # --- Fit dipoles to calibration post-stim trials ---
    print("\nFitting dipoles to calibration trials...")
    for orientation_label, orientation in [('fixed', 'use_fitted'), ('free', None)]:
        dipoles_for_trials, extraction_times = dipole_fitter.fit_trials(
            cal_post_epochs, orientation=orientation)
        mean_time_ms = np.mean(extraction_times) * 1e3
        print(f"  {orientation_label} orientation: {len(dipoles_for_trials)} dipoles, "
              f"avg extraction time {mean_time_ms:.2f} ms")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print(f"  Calibration params obtained: {list(calibrator.calibration_params.keys())}")
    print(f"  Dipole position index: {position_index}")
    print(f"  Dipole fit window: [{tmin_fit*1000:.1f}, {tmax_fit*1000:.1f}] ms")
    print(f"  Pre-stim epochs for classifier: {cal_pre_epochs.get_data(copy=False).shape}")
    print(f"  Post-stim epochs (dipole-fitted): {cal_post_epochs.get_data(copy=False).shape}")
    print("=" * 70)

    # =========================================================================
    # REFERENCE: offline pipeline prediction for the first non-calibration trial
    #
    # The .npz file contains predictions from the fully offline pipeline
    # (train_transfer.py) for the intervention (post-calibration) phase only —
    # calibration trials are excluded, so index 0 is the first trial after the
    # N_CALIBRATION_TRIALS boundary.
    #
    # This is what the online path should eventually reproduce for each trial.
    # (Missing code parts: feature extraction → model inference → decision.)
    # =========================================================================
    OFFLINE_PREDICTIONS_PATH = (
        Path(__file__).resolve().parent
        / "results/replicate_prime/2026-05-24_09-34-12"
        / f"predictions_subj_18_fold_2.npz"
    )
    print("\n" + "=" * 70)
    print("OFFLINE REFERENCE (first non-calibration trial)")
    print("=" * 70)
    offline = np.load(OFFLINE_PREDICTIONS_PATH)
    first_pred = offline["predictions"][0]
    first_actual = offline["actual_values"][0]
    print(f"  Raw prediction (sigmoid prob): {first_pred:.6f}")
    print(f"  Actual value (soft label):     {first_actual:.6f}")
    print(f"  Hard label (>0.5):             {int(first_actual > 0.5)}")
    print(f"  Hard prediction (>0.5):        {int(first_pred > 0.5)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
