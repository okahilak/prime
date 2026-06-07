#%%
import time
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np

from prime.prime_config import (
    get_calibration_time_range,
    get_post_time_range,
    get_processed_sfreq,
    get_raw_sfreq,
)
from prime.online_preprocessing.preprocessor import (
    Preprocessor,
    crop_eeg_buffer,
    crop_mne_trial_to_buffer,
)
from prime.online_preprocessing.trial_loader import TrialLoader
from prime.online_preprocessing.utils.resampling import resample_buffer_polyphase

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

N_TRIALS_CALIBRATE = 125

mne.set_log_level("ERROR")


# ==================== Single-Trial Worker ====================

BUFFER_KEYS = ('qc', 'model', 'post', 'dipole', 'post_raw')


def extract_post_raw(eeg_buffer, relative_timestamps):
    """Crop and resample the post window without further preprocessing."""
    post_tmin, post_tmax = get_post_time_range()
    post_buffer = crop_eeg_buffer(eeg_buffer, relative_timestamps, post_tmin, post_tmax)
    post_buffer = resample_buffer_polyphase(
        post_buffer,
        sfreq_from=get_raw_sfreq(),
        sfreq_to=get_processed_sfreq(),
    )
    return np.ascontiguousarray(post_buffer.T, dtype=np.float64)


def process_single_trial(eeg_buffer, relative_timestamps, preprocessor):
    """Process a single trial. Returns a buffers dict or None if rejected."""
    buffers_pre = preprocessor.preprocess_pre(eeg_buffer, relative_timestamps)
    buffers_post = preprocessor.preprocess_post(eeg_buffer, relative_timestamps)
    if buffers_pre is None or buffers_post is None:
        return None
    return {
        **buffers_pre,
        **buffers_post,
        'post_raw': extract_post_raw(eeg_buffer, relative_timestamps),
    }


_online_trial_worker_state = {}


def _init_online_trial_worker(
    batch_data, info, tmin, events, preprocessor,
    trial_tmin, trial_tmax,
):
    mne.set_log_level("ERROR")
    global _online_trial_worker_state
    _online_trial_worker_state = {
        'batch_data': batch_data,
        'info': info,
        'tmin': tmin,
        'events': events,
        'preprocessor': preprocessor,
        'trial_tmin': trial_tmin,
        'trial_tmax': trial_tmax,
    }


def _process_online_trial_worker(trial_idx):
    s = _online_trial_worker_state
    trial = mne.EpochsArray(
        s['batch_data'][trial_idx:trial_idx + 1], info=s['info'],
        events=s['events'][trial_idx:trial_idx + 1], tmin=s['tmin'], verbose=False)
    eeg_buffer, relative_timestamps = crop_mne_trial_to_buffer(
        trial, s['trial_tmin'], s['trial_tmax'],
    )
    processed = process_single_trial(eeg_buffer, relative_timestamps, s['preprocessor'])
    return trial_idx, processed


# ==================== Calibration persistence (two-step simulation) ====================

def _save_calibration_bundle(path, calibration_params):
    np.save(path, calibration_params, allow_pickle=True)


def _load_calibration_bundle(path):
    return np.load(path, allow_pickle=True).item()


def _run_calibration_stage(trial_loader, forward_path, calibration_bundle_path):
    """Calibration only: writes bundle to disk; no state returned except the path."""
    print("Appending calibration trials...")

    trial_tmin, trial_tmax = get_calibration_time_range()
    preprocessor = Preprocessor(forward_path)
    for trial_idx in range(N_TRIALS_CALIBRATE):
        eeg_buffer, relative_timestamps = crop_mne_trial_to_buffer(
            trial_loader.get_trial(trial_idx),
            trial_tmin, trial_tmax,
        )
        preprocessor.add_trial(eeg_buffer, relative_timestamps)

    print("Calibrating...")

    start_time = time.time()
    buffers = preprocessor.calibrate()

    _save_calibration_bundle(calibration_bundle_path, preprocessor.calibration_params)

    subject_output = Path(calibration_bundle_path).parent
    subject_id = Path(calibration_bundle_path).stem.removesuffix('_calibration_bundle')
    _save_buffers(subject_output, subject_id, 'calibration', buffers)

    print(f"Used {len(buffers['model'])} trials for calibration")

    end_time = time.time()
    print(f"Calibration stage took {end_time - start_time:.2f} seconds")


def _save_buffers(subject_output, subject_id, label, buffers):
    for key, data in buffers.items():
        if data.ndim != 3:
            raise ValueError(
                f"expected 3D buffer '{key}' (n_trials, n_channels, n_times), got shape {data.shape}"
            )
        np.save(
            os.path.join(subject_output, f"{subject_id}_{label}_{key}_buffer.npy"),
            data,
        )


def _process_and_save_trial_group(
    epochs_data, events, info, tmin, preprocessor, subject_output, subject_id, label,
    trial_tmin, trial_tmax,
):
    """Batch-process a pre-indexed group of trials and save buffer arrays to disk."""
    n_trials = epochs_data.shape[0]
    buffer_lists = {key: [] for key in BUFFER_KEYS}
    bad_trials = []

    initargs = (
        epochs_data, info, tmin, events, preprocessor,
        trial_tmin, trial_tmax,
    )
    with ProcessPoolExecutor(
        max_workers=4,
        initializer=_init_online_trial_worker,
        initargs=initargs,
    ) as executor:
        results = list(executor.map(_process_online_trial_worker, range(n_trials)))

    for i, (_, processed) in enumerate(results):
        if processed is not None:
            for key in BUFFER_KEYS:
                buffer_lists[key].append(processed[key])
        else:
            bad_trials.append(i)

    buffers = {}
    for key in BUFFER_KEYS:
        if buffer_lists[key]:
            buffers[key] = np.stack(buffer_lists[key], axis=0)
        else:
            buffers[key] = np.empty((0, 0, 0), dtype=np.float64)
    _save_buffers(subject_output, subject_id, label, buffers)


def _run_online_processing_stage(
    trial_loader, subject_output, subject_id, calibration_bundle_path,
    forward_path,
):
    """Online trial processing: only trial_loader and calibration bundle from disk."""
    start_time = time.time()

    calibration_params = _load_calibration_bundle(calibration_bundle_path)
    epochs = trial_loader._epochs
    trial_tmin, trial_tmax = get_calibration_time_range()
    preprocessor = Preprocessor.from_bundle(calibration_params, forward_path)
    epochs_data = trial_loader._eeg_data

    _process_and_save_trial_group(
        epochs_data[:N_TRIALS_CALIBRATE], epochs.events[:N_TRIALS_CALIBRATE],
        epochs.info, epochs.tmin,
        preprocessor, subject_output, subject_id, 'calibration',
        trial_tmin, trial_tmax,
    )
    _process_and_save_trial_group(
        epochs_data[N_TRIALS_CALIBRATE:], epochs.events[N_TRIALS_CALIBRATE:],
        epochs.info, epochs.tmin,
        preprocessor, subject_output, subject_id, 'intervention',
        trial_tmin, trial_tmax,
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
