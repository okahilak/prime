#%%
import time
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np

from prime.prime_config import get_raw_post_time_range, get_raw_pre_time_range
from prime.online_preprocessing.preprocessor import (
    Preprocessor,
    crop_mne_trial_to_raw_epochs,
)
from prime.online_preprocessing.trial_loader import TrialLoader

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

N_TRIALS_CALIBRATE = 125

mne.set_log_level("ERROR")


# ==================== Single-Trial Worker ====================

def process_single_trial(raw_pre, raw_post, preprocessor):
    """Process a single trial. Returns (epoch_pre, epoch_post) or None if rejected."""
    epoch_pre = preprocessor.preprocess_pre(raw_pre)
    epoch_post = preprocessor.preprocess_post(raw_post)
    if epoch_pre is None or epoch_post is None:
        return None
    return epoch_pre, epoch_post


_online_trial_worker_state = {}


def _init_online_trial_worker(
    batch_data, info, tmin, events, preprocessor,
    raw_pre_tmin, raw_pre_tmax, raw_post_tmin, raw_post_tmax,
):
    mne.set_log_level("ERROR")
    global _online_trial_worker_state
    _online_trial_worker_state = {
        'batch_data': batch_data,
        'info': info,
        'tmin': tmin,
        'events': events,
        'preprocessor': preprocessor,
        'raw_pre_tmin': raw_pre_tmin,
        'raw_pre_tmax': raw_pre_tmax,
        'raw_post_tmin': raw_post_tmin,
        'raw_post_tmax': raw_post_tmax,
    }


def _process_online_trial_worker(trial_idx):
    s = _online_trial_worker_state
    trial = mne.EpochsArray(
        s['batch_data'][trial_idx:trial_idx + 1], info=s['info'],
        events=s['events'][trial_idx:trial_idx + 1], tmin=s['tmin'], verbose=False)
    raw_pre, raw_post = crop_mne_trial_to_raw_epochs(
        trial, s['raw_pre_tmin'], s['raw_pre_tmax'], s['raw_post_tmin'], s['raw_post_tmax'],
    )
    processed = process_single_trial(raw_pre, raw_post, s['preprocessor'])
    return trial_idx, processed


# ==================== Calibration persistence (two-step simulation) ====================

def _save_calibration_bundle(path, calibration_params):
    np.save(path, calibration_params, allow_pickle=True)


def _load_calibration_bundle(path):
    return np.load(path, allow_pickle=True).item()


def _run_calibration_stage(trial_loader, forward_path, calibration_bundle_path):
    """Calibration only: writes bundle to disk; no state returned except the path."""
    print("Appending calibration trials...")

    raw_pre_tmin, raw_pre_tmax = get_raw_pre_time_range()
    raw_post_tmin, raw_post_tmax = get_raw_post_time_range()
    preprocessor = Preprocessor(forward_path)
    for trial_idx in range(N_TRIALS_CALIBRATE):
        raw_pre, raw_post = crop_mne_trial_to_raw_epochs(
            trial_loader.get_trial(trial_idx),
            raw_pre_tmin, raw_pre_tmax, raw_post_tmin, raw_post_tmax,
        )
        preprocessor.add_raw_pre(raw_pre)
        preprocessor.add_raw_post(raw_post)

    print("Calibrating...")

    start_time = time.time()
    cal_pre, _cal_post = preprocessor.calibrate()

    _save_calibration_bundle(calibration_bundle_path, preprocessor.calibration_params)

    print(f"Used {len(cal_pre)} trials for calibration")

    end_time = time.time()
    print(f"Calibration stage took {end_time - start_time:.2f} seconds")


def _process_and_save_trial_group(
    epochs_data, events, info, tmin, preprocessor, subject_output, subject_id, label,
    raw_pre_tmin, raw_pre_tmax, raw_post_tmin, raw_post_tmax,
):
    """Batch-process a pre-indexed group of trials and save pre/post epochs to disk."""
    n_trials = epochs_data.shape[0]
    pre_list = []
    post_list = []
    bad_trials = []

    initargs = (
        epochs_data, info, tmin, events, preprocessor,
        raw_pre_tmin, raw_pre_tmax, raw_post_tmin, raw_post_tmax,
    )
    with ProcessPoolExecutor(
        max_workers=4,
        initializer=_init_online_trial_worker,
        initargs=initargs,
    ) as executor:
        results = list(executor.map(_process_online_trial_worker, range(n_trials)))

    for i, (_, processed) in enumerate(results):
        if processed is not None:
            epoch_pre, epoch_post = processed
            pre_list.append(epoch_pre)
            post_list.append(epoch_post)
        else:
            bad_trials.append(i)

    epochs_pre_final = mne.concatenate_epochs(pre_list)
    epochs_post_final = mne.concatenate_epochs(post_list)

    for epoch, segment in zip([epochs_pre_final, epochs_post_final], ['pre', 'post']):
        epoch.save(os.path.join(subject_output, f"{subject_id}_{label}_{segment}.fif"), overwrite=True)


def _run_online_processing_stage(
    trial_loader, subject_output, subject_id, calibration_bundle_path,
    forward_path,
):
    """Online trial processing: only trial_loader and calibration bundle from disk."""
    start_time = time.time()

    calibration_params = _load_calibration_bundle(calibration_bundle_path)
    epochs = trial_loader._epochs
    raw_pre_tmin, raw_pre_tmax = get_raw_pre_time_range()
    raw_post_tmin, raw_post_tmax = get_raw_post_time_range()
    preprocessor = Preprocessor.from_bundle(calibration_params, forward_path)
    epochs_data = trial_loader._eeg_data

    _process_and_save_trial_group(
        epochs_data[:N_TRIALS_CALIBRATE], epochs.events[:N_TRIALS_CALIBRATE],
        epochs.info, epochs.tmin,
        preprocessor, subject_output, subject_id, 'calibration',
        raw_pre_tmin, raw_pre_tmax, raw_post_tmin, raw_post_tmax,
    )
    _process_and_save_trial_group(
        epochs_data[N_TRIALS_CALIBRATE:], epochs.events[N_TRIALS_CALIBRATE:],
        epochs.info, epochs.tmin,
        preprocessor, subject_output, subject_id, 'intervention',
        raw_pre_tmin, raw_pre_tmax, raw_post_tmin, raw_post_tmax,
    )

    end_time = time.time()
    print(f"Online processing stage took {end_time - start_time:.2f} seconds")


def run_subject_processing(subject_id: str):
    """Main preprocessing pipeline for a single subject."""
    output_path = DATA_ROOT / "processed"
    subject_output = output_path / subject_id
    os.makedirs(subject_output, exist_ok=True)

    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
    calibration_bundle_path = subject_output / f'{subject_id}_calibration_bundle.npy'

    trial_loader = TrialLoader(subject_id)
    _run_calibration_stage(trial_loader, forward_path, calibration_bundle_path)

    _run_online_processing_stage(
        trial_loader, subject_output, subject_id, calibration_bundle_path, forward_path)

    print("Done")


# %%
def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for a single subject.")
    parser.add_argument("--subject", required=True, type=str, help="Subject identifier.")
    args = parser.parse_args()
    run_subject_processing(subject_id=args.subject)

if __name__ == "__main__":
    main()
