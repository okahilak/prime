#%%
import time
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np

try:
    from .config import get_default_config
    from .calibrator import Calibrator, ProcessedTrial
    from .trial_loader import TrialLoader
except ImportError:
    from config import get_default_config
    from calibrator import Calibrator, ProcessedTrial
    from trial_loader import TrialLoader

DATA_ROOT = Path("~/prime-data").expanduser()

N_TRIALS_CALIBRATE = 125

mne.set_log_level("ERROR")


# ==================== Single-Trial Worker ====================

def process_single_trial(trial, calibrator) -> ProcessedTrial | None:
    """Process a single trial. Returns ProcessedTrial or None if rejected."""
    return calibrator.preprocess(trial)


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
    processed = process_single_trial(trial, s['calibrator'])
    return trial_idx, processed


# ==================== Calibration persistence (two-step simulation) ====================

def _save_calibration_bundle(path, calibration_params):
    np.save(path, calibration_params, allow_pickle=True)


def _load_calibration_bundle(path):
    return np.load(path, allow_pickle=True).item()


def _run_calibration_stage(trial_loader, cfg, forward_path, calibration_bundle_path):
    """Calibration only: writes bundle to disk; no state returned except the path."""
    print("Appending calibration trials...")

    calibrator = Calibrator(cfg, forward_path)
    for trial_idx in range(N_TRIALS_CALIBRATE):
        calibrator.add_raw_trial(trial_loader.get_trial(trial_idx))

    print("Calibrating...")

    start_time = time.time()
    cal_trials = calibrator.calibrate()

    _save_calibration_bundle(calibration_bundle_path, calibrator.calibration_params)

    print(f"Used {len(cal_trials)} trials for calibration")

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

    for i, (_, processed) in enumerate(results):
        if processed is not None:
            pre_list.append(processed.epoch_pre)
            post_list.append(processed.epoch_post)
        else:
            bad_trials.append(i)

    epochs_pre_final = mne.concatenate_epochs(pre_list)
    epochs_post_final = mne.concatenate_epochs(post_list)

    for epoch, segment in zip([epochs_pre_final, epochs_post_final], ['pre', 'post']):
        epoch.save(os.path.join(subject_output, f"{subject_id}_{label}_{segment}.fif"), overwrite=True)


def _run_online_processing_stage(
    trial_loader, cfg, subject_output, subject_id, calibration_bundle_path,
    forward_path,
):
    """Online trial processing: only config, trial_loader, and calibration bundle from disk."""
    start_time = time.time()

    calibration_params = _load_calibration_bundle(calibration_bundle_path)
    calibrator = Calibrator.from_bundle(cfg, calibration_params, forward_path)
    epochs_data = trial_loader._eeg_data
    epochs = trial_loader._epochs

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

    trial_loader = TrialLoader(subject_id, cfg)
    _run_calibration_stage(trial_loader, cfg, forward_path, calibration_bundle_path)

    cfg = get_default_config()
    _run_online_processing_stage(
        trial_loader, cfg, subject_output, subject_id, calibration_bundle_path, forward_path)

    print("Done")


# %%
def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for a single subject.")
    parser.add_argument("--subject", required=True, type=str, help="Subject identifier.")
    args = parser.parse_args()
    run_subject_processing(subject_id=args.subject)

if __name__ == "__main__":
    main()
