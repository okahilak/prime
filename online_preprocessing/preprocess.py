#%%
import time
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np

from config import get_default_config
from calibrator import (
    Calibrator,
    _single_trial_epochs_from_arrays,
)

DATA_ROOT = Path("~/prime-data").expanduser()

N_TRIALS_CALIBRATE = 125

mne.set_log_level("ERROR")


# ==================== Single-Trial Worker ====================

def process_single_trial(trial, calibrator):
    """Process a single trial. Returns (success, epoch_pre, epoch_post)."""
    result_pre = calibrator.preprocess_pre(trial)
    result_post = calibrator.preprocess_post(trial)
    if result_pre is False or result_post is False:
        return False, None, None
    return True, result_pre, result_post


_online_trial_worker_state = {}


def _init_online_trial_worker(batch_data, info, tmin, events, calibrator):
    mne.set_log_level("ERROR")
    global _online_trial_worker_state
    _online_trial_worker_state = {
        'batch_data': batch_data,
        'info': info,
        'tmin': tmin,
        'events': events,
        'calibrator': calibrator,
    }


def _process_online_trial_worker(trial_idx):
    s = _online_trial_worker_state
    trial = mne.EpochsArray(
        s['batch_data'][trial_idx:trial_idx + 1], info=s['info'],
        events=s['events'][trial_idx:trial_idx + 1], tmin=s['tmin'], verbose=False)
    success, result_pre, result_post = process_single_trial(trial, s['calibrator'])
    return trial_idx, success, result_pre, result_post


# ==================== Calibration persistence (two-step simulation) ====================

def _save_calibration_bundle(path, calibration_params):
    np.save(path, calibration_params, allow_pickle=True)


def _load_calibration_bundle(path):
    return np.load(path, allow_pickle=True).item()


def _load_subject_epochs(subject_id, cfg):
    print("Loading data...")

    data_path = DATA_ROOT / "raw"
    subject_path = data_path / subject_id
    if not subject_path.exists():
        raise FileNotFoundError(f"Subject directory not found at {subject_path}")

    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
    if not forward_path.exists():
        raise FileNotFoundError(f"Forward solution not found at {forward_path}. Run: python {DATA_ROOT / 'build_fsaverage_forward.py'}")
    channel_order = mne.read_forward_solution(forward_path).ch_names
    montage = mne.channels.make_standard_montage('standard_1005')

    epochs = mne.read_epochs_eeglab(os.path.join(subject_path, f'{subject_id}_task-tep_all_eeg.set'))
    epochs.pick(cfg.common_channels)
    epochs.reorder_channels(channel_order)
    epochs.set_montage(None)
    epochs.set_montage(montage)

    if channel_order != epochs.ch_names:
        raise ValueError(f"Channel order mismatch: {channel_order} vs {epochs.ch_names}")

    return epochs


def _run_calibration_stage(epochs, cfg, forward_path, calibration_bundle_path):
    """Calibration only: writes bundle to disk; no state returned except the path."""
    all_eeg_data = epochs.get_data(copy=False)
    all_events = epochs.events

    print("Appending calibration trials...")

    calibrator = Calibrator(cfg, forward_path)
    for trial_idx in range(N_TRIALS_CALIBRATE):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        calibrator.add_trial(trial)

    print("Calibrating...")

    start_time = time.time()
    n_successful_trials = calibrator.calibrate()

    _save_calibration_bundle(calibration_bundle_path, calibrator.calibration_params)

    print(f"Used {n_successful_trials} trials for calibration")

    end_time = time.time()
    print(f"Calibration stage took {end_time - start_time:.2f} seconds")


def _process_and_save_trial_group(
    epochs_data, events, info, tmin, calibrator, subject_output, subject_id, label
):
    """Batch-process a pre-indexed group of trials and save pre/post epochs to disk."""
    n_trials = epochs_data.shape[0]
    pre_list = []
    post_list = []
    bad_trials = []

    initargs = (epochs_data, info, tmin, events, calibrator)
    with ProcessPoolExecutor(
        max_workers=4,
        initializer=_init_online_trial_worker,
        initargs=initargs,
    ) as executor:
        results = list(executor.map(_process_online_trial_worker, range(n_trials)))

    for i, (_, success, result_pre, result_post) in enumerate(results):
        if success:
            pre_list.append(result_pre)
            post_list.append(result_post)
        else:
            bad_trials.append(i)

    epochs_pre_final = mne.concatenate_epochs(pre_list)
    epochs_post_final = mne.concatenate_epochs(post_list)

    for epoch, segment in zip([epochs_pre_final, epochs_post_final], ['pre', 'post']):
        epoch.save(os.path.join(subject_output, f"{subject_id}_{label}_{segment}.fif"), overwrite=True)


def _run_online_processing_stage(
    epochs, cfg, subject_output, subject_id, calibration_bundle_path,
    forward_path,
):
    """Online trial processing: only config, epochs, and calibration bundle from disk."""
    start_time = time.time()

    calibration_params = _load_calibration_bundle(calibration_bundle_path)
    calibrator = Calibrator.from_bundle(cfg, calibration_params, forward_path)
    epochs_data = epochs.get_data(copy=False)

    _process_and_save_trial_group(
        epochs_data[:N_TRIALS_CALIBRATE], epochs.events[:N_TRIALS_CALIBRATE],
        epochs.info, epochs.tmin,
        calibrator, subject_output, subject_id, label='calibration',
    )
    _process_and_save_trial_group(
        epochs_data[N_TRIALS_CALIBRATE:], epochs.events[N_TRIALS_CALIBRATE:],
        epochs.info, epochs.tmin,
        calibrator, subject_output, subject_id, label='intervention',
    )

    end_time = time.time()
    print(f"Online processing stage took {end_time - start_time:.2f} seconds")


def run_subject_processing(subject_id: str):
    """Main preprocessing pipeline for a single subject."""
    cfg = get_default_config()
    output_path = DATA_ROOT / "processed"
    subject_output = output_path / subject_id
    os.makedirs(subject_output, exist_ok=True)

    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
    calibration_bundle_path = subject_output / f'{subject_id}_calibration_bundle.npy'

    epochs = _load_subject_epochs(subject_id, cfg)
    _run_calibration_stage(epochs, cfg, forward_path, calibration_bundle_path)

    cfg = get_default_config()
    _run_online_processing_stage(
        epochs, cfg, subject_output, subject_id, calibration_bundle_path, forward_path)

    print("Done")


# %%
def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for a single subject.")
    parser.add_argument("--subject", required=True, type=str, help="Subject identifier.")
    args = parser.parse_args()
    run_subject_processing(subject_id=args.subject)

if __name__ == "__main__":
    main()
