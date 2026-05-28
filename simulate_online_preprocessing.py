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
import pandas as pd

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
from TMS_EEG_moabb import TEPNormalizer

# =============================================================================
# Hard-coded constants
# =============================================================================
SUBJECT_ID = "sub-021"
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

    # --- Process calibration trials: pre + post together (offline requires both) ---
    print("\nPreprocessing calibration trials (both pre & post)...")
    cal_pre_list = []
    cal_post_list = []
    for trial_idx in range(N_CALIBRATION_TRIALS):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        result_pre = calibrator.preprocess_pre(trial)
        result_post = calibrator.preprocess_post(trial)
        if result_pre is not False and result_post is not False:
            cal_pre_list.append(result_pre)
            cal_post_list.append(result_post)

    print(f"  {len(cal_post_list)}/{N_CALIBRATION_TRIALS} calibration trials survived "
          f"(both pre & post)")
    cal_pre_epochs = mne.concatenate_epochs(cal_pre_list)
    cal_post_epochs = mne.concatenate_epochs(cal_post_list)

    # --- Calibrate dipole fitting parameters ---
    print("\nCalibrating dipole parameters...")
    dipole_fitter = DipoleFitter(forward_path)
    dipole_fitter.fit(cal_post_epochs)
    position_index = dipole_fitter.fitting_info['position_index']
    tmin_fit, tmax_fit = dipole_fitter.fitting_info['time_range']
    print(f"  Position index: {position_index}")
    print(f"  Time range: [{tmin_fit*1000:.1f}, {tmax_fit*1000:.1f}] ms")

    # --- Fit dipoles to calibration post-stim trials ---
    print("\nFitting dipoles to calibration trials...")
    cal_dipoles_free = None
    for orientation_label, orientation in [('fixed', 'use_fitted'), ('free', None)]:
        dipoles_for_trials, extraction_times = dipole_fitter.fit_trials(
            cal_post_epochs, orientation=orientation)
        mean_time_ms = np.mean(extraction_times) * 1e3
        print(f"  {orientation_label} orientation: {len(dipoles_for_trials)} dipoles, "
              f"avg extraction time {mean_time_ms:.2f} ms")
        if orientation is None:
            cal_dipoles_free = dipoles_for_trials

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
    # INTERVENTION PHASE — process post-calibration trials, fit dipoles,
    # compute normalized TEP labels using the same Calibrator preprocessing
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERVENTION PHASE — TEP label computation")
    print("=" * 70)

    n_intervention = n_total_trials - N_CALIBRATION_TRIALS
    print(f"Processing {n_intervention} intervention trials...")

    # --- Process intervention trials: require both pre & post to pass ---
    int_post_list = []
    int_pre_list = []
    for trial_idx in range(N_CALIBRATION_TRIALS, n_total_trials):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        result_pre = calibrator.preprocess_pre(trial)
        result_post = calibrator.preprocess_post(trial)
        if result_pre is not False and result_post is not False:
            int_pre_list.append(result_pre)
            int_post_list.append(result_post)
        if (trial_idx - N_CALIBRATION_TRIALS + 1) % 100 == 0:
            print(f"  Preprocessed {trial_idx - N_CALIBRATION_TRIALS + 1}/{n_intervention} "
                  f"intervention trials")

    print(f"  {len(int_post_list)}/{n_intervention} intervention trials survived "
          f"(both pre & post)")
    int_post_epochs = mne.concatenate_epochs(int_post_list)

    print("\nFitting free-orientation dipoles to intervention trials...")
    int_dipoles_free, int_extraction_times = dipole_fitter.fit_trials(
        int_post_epochs, orientation=None)
    print(f"  {len(int_dipoles_free)} dipoles fitted, "
          f"avg time {np.mean(int_extraction_times)*1e3:.2f} ms")

    # --- TEP normalization (matching offline path) ---
    cal_tep_amplitudes = np.array([d['amplitude'] for d in cal_dipoles_free]).flatten()
    int_tep_amplitudes = np.array([d['amplitude'] for d in int_dipoles_free]).flatten()
    all_tep_amplitudes = np.concatenate([cal_tep_amplitudes, int_tep_amplitudes])
    period_labels = (['calibration'] * len(cal_tep_amplitudes) +
                     ['intervention'] * len(int_tep_amplitudes))

    full_metadata = pd.DataFrame({
        'TEP_amplitude': all_tep_amplitudes,
        'period': period_labels,
    })
    cal_metadata = full_metadata[full_metadata['period'] == 'calibration']

    # Fit normalizer on calibration, transform full sequence
    normalizer = TEPNormalizer(target_col='TEP_amplitude', scale_factor=1.0)
    normalizer.fit(cal_metadata)
    all_labels = normalizer.transform(full_metadata)

    # Extract intervention labels
    int_labels = all_labels[len(cal_tep_amplitudes):]
    print(f"\n  Calibration TEP amplitudes: {len(cal_tep_amplitudes)}")
    print(f"  Intervention TEP amplitudes: {len(int_tep_amplitudes)}")
    print(f"  Intervention labels (first 5): {int_labels[:5]}")

    # =========================================================================
    # COMPARE WITH OFFLINE LABELS
    # =========================================================================
    OFFLINE_PREDICTIONS_PATH = (
        Path(__file__).resolve().parent
        / "results/2026-05-28_13-26-46_replicate_prime"
        / f"predictions_subj_21_fold_1.npz"
    )
    print("\n" + "=" * 70)
    print("COMPARE ONLINE vs OFFLINE LABELS")
    print("=" * 70)
    offline = np.load(OFFLINE_PREDICTIONS_PATH)
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


if __name__ == "__main__":
    main()
