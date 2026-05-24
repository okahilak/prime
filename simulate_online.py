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
from online_preprocessing.preprocess import (
    preprocess_pre_trial,
    preprocess_post_trial,
    _compute_leadfield,
    _load_subject_epochs,
    _single_trial_epochs_from_arrays,
)
from online_preprocessing.calibrate_dipole import (
    dipoles_for_times,
    determine_optimal_time_range,
    determine_optimal_ori_and_pos,
)
from online_preprocessing.fit_dipole import fit_dipoles_to_single_trials

# =============================================================================
# Hard-coded constants
# =============================================================================
SUBJECT_ID = "sub-018"
N_CALIBRATION_TRIALS = 125
DATA_ROOT = Path("~/prime-data").expanduser()


def load_forward(cfg):
    """Load the fsaverage forward solution with common channels."""
    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
    forward = mne.read_forward_solution(str(forward_path), verbose=False)
    forward = forward.pick_channels(cfg.common_channels, ordered=True)
    return forward


def main():
    print("=" * 70)
    print("SIMULATE ONLINE PROCESSING (trial-by-trial)")
    print(f"Subject: {SUBJECT_ID}")
    print(f"Calibration trials: {N_CALIBRATION_TRIALS}")
    print("=" * 70)

    # --- Setup ---
    cfg = get_default_config()
    leadfield = _compute_leadfield()
    forward = load_forward(cfg)

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
    calibrator = Calibrator(cfg)

    for trial_idx in range(N_CALIBRATION_TRIALS):
        # Simulate receiving a single trial
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)

        calibrator.add_trial(trial)

        if (trial_idx + 1) % 25 == 0:
            print(f"  Accumulated {trial_idx + 1}/{N_CALIBRATION_TRIALS} calibration trials")

    # --- We have enough trials: run calibration ---
    print("\nRunning calibration preprocessing...")
    t0 = time.time()
    calibration_params, n_successful_trials = calibrator.calibrate(leadfield)
    print(f"Calibration preprocessing done in {time.time() - t0:.2f}s, "
          f"used {n_successful_trials} trials")

    # --- Calibrate dipole from calibration post-stim data ---
    # First, preprocess each calibration trial's post-stim to get clean epochs for dipole calibration
    print("\nPreprocessing calibration trials (post-stim) for dipole calibration...")
    cal_post_list = []
    for trial_idx in range(N_CALIBRATION_TRIALS):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        epoch_post = trial.copy().crop(cfg.post_range[0], cfg.post_range[1])
        epoch_post.resample(cfg.target_sfreq, method='polyphase')

        result_post = preprocess_post_trial(epoch_post, calibration_params, cfg)
        if result_post is not False:
            cal_post_list.append(result_post)

    print(f"  {len(cal_post_list)}/{N_CALIBRATION_TRIALS} post-stim trials survived preprocessing")
    cal_post_epochs = mne.concatenate_epochs(cal_post_list)

    # --- Calibrate dipole fitting parameters ---
    print("\nCalibrating dipole parameters...")
    evoked = cal_post_epochs.copy().average()

    tmin_init, tmax_init = 0.038, 0.050
    min_window_size = 3
    max_window_size = 6
    window_size_exponent = 1.5

    best_dipole_per_time = dipoles_for_times(evoked, forward, tmin_init, tmax_init)
    time_range, dipoles_in_time_range = determine_optimal_time_range(
        best_dipole_per_time, min_window_size, max_window_size, window_size_exponent)
    position_index, orientation = determine_optimal_ori_and_pos(dipoles_in_time_range, forward)

    tmin_fit, tmax_fit = float(time_range[0]), float(time_range[1])
    print(f"  Position index: {position_index}")
    print(f"  Time range: [{tmin_fit*1000:.1f}, {tmax_fit*1000:.1f}] ms")

    # --- Preprocess calibration trials (pre-stim) for classifier input ---
    print("\nPreprocessing calibration trials (pre-stim) for classifier...")
    cal_pre_list = []
    for trial_idx in range(N_CALIBRATION_TRIALS):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        epoch_pre = trial.copy().crop(cfg.pre_range[0], cfg.pre_range[1])
        epoch_pre.resample(cfg.target_sfreq, method='polyphase')

        result_pre = preprocess_pre_trial(epoch_pre, calibration_params, cfg)
        if result_pre is not False:
            cal_pre_list.append(result_pre)

    print(f"  {len(cal_pre_list)}/{N_CALIBRATION_TRIALS} pre-stim trials survived preprocessing")
    cal_pre_epochs = mne.concatenate_epochs(cal_pre_list)

    # --- Fit dipoles to calibration post-stim trials ---
    print("\nFitting dipoles to calibration trials...")
    for fixed_ori in [orientation, None]:
        ori_label = "fixed" if fixed_ori is not None else "free"
        dipoles_for_trials, extraction_times = fit_dipoles_to_single_trials(
            cal_post_epochs, forward, position_index, fixed_ori, tmin_fit, tmax_fit)
        mean_time_ms = np.mean(extraction_times) * 1e3
        print(f"  {ori_label} orientation: {len(dipoles_for_trials)} dipoles, "
              f"avg extraction time {mean_time_ms:.2f} ms")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print(f"  Calibration params obtained: {list(calibration_params.keys())}")
    print(f"  Dipole position index: {position_index}")
    print(f"  Dipole fit window: [{tmin_fit*1000:.1f}, {tmax_fit*1000:.1f}] ms")
    print(f"  Pre-stim epochs for classifier: {cal_pre_epochs.get_data(copy=False).shape}")
    print(f"  Post-stim epochs (dipole-fitted): {cal_post_epochs.get_data(copy=False).shape}")
    print("=" * 70)


if __name__ == "__main__":
    main()
