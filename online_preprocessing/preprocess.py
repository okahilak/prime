#%%
import time
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import mne
import numpy as np
from scipy.stats import median_abs_deviation, zscore

from utils.ssp_sir_python import ssp_sir_single_trial
from config import get_default_config
from calibrator import (
    Calibrator,
    _apply_filter,
    _interpolate_bad_channels,
    _single_trial_epochs_from_arrays,
    _compute_leadfield,
)

DATA_ROOT = Path("~/prime-data").expanduser()

N_TRIALS_CALIBRATE = 125

mne.set_log_level("ERROR")


# ==================== Single-Trial Processing ====================

def preprocess_pre_trial(epoch_pre, calibration_params, cfg):
    """Preprocess a single pre-stimulus trial using calibrated parameters."""
    dicts = cfg.to_dicts()
    trial_reject_opts = dicts['trial_reject_opts']
    filter_opts = dicts['filter_opts']

    # Interpolate bad channels
    if calibration_params['bad_channels']:
        epoch_pre, _ = _interpolate_bad_channels(
            epoch_pre, calibration_params['bad_channels'], calibration_params['channel_interpolation_info'])

    if cfg.use_ica_on_pre:
        raise NotImplementedError("Applying ICA to pre-stimulus in single-trials is not currently implemented")

    data = epoch_pre.get_data(copy=True)

    # Apply filter
    data = _apply_filter(
        data, calibration_params['pre_stim_filter'], filter_opts['pad_time'], epoch_pre.info['sfreq'])

    # Mean subtraction (average reference)
    data -= np.mean(data, axis=1)

    # Global MAD check
    mad_val = median_abs_deviation(data, axis=(1, 2))[0]
    z_mad = (mad_val - calibration_params['good_trial_stats_pre']['mads_mean']) / calibration_params['good_trial_stats_pre']['mads_std']

    threshold = trial_reject_opts['pre']['global_zscore_threshold']
    if z_mad < threshold[0] or z_mad > threshold[1]:
        return False

    # Local MAD check
    local_mad = median_abs_deviation(data, axis=2)
    z_local = zscore(local_mad, axis=1)

    local_threshold = trial_reject_opts['pre']['local_zscore_threshold']
    if np.any(np.abs(z_local) > local_threshold):
        return False

    # Rebuild epoch
    epoch_pre = mne.EpochsArray(data, info=epoch_pre.info, events=epoch_pre.events, tmin=epoch_pre.times[0], verbose=False)
    return epoch_pre


def preprocess_post_trial(epoch_post, calibration_params, cfg):
    """Preprocess a single post-stimulus trial using calibrated parameters."""
    dicts = cfg.to_dicts()
    trial_reject_opts = dicts['trial_reject_opts']

    # Baseline correction
    epoch_post.apply_baseline(cfg.baseline)

    # First artifact interpolation
    epoch_post = mne.preprocessing.fix_stim_artifact(
        epoch_post, tmin=cfg.artifact_window_1[0], tmax=cfg.artifact_window_1[1], mode='window')

    # Interpolate bad channels
    if calibration_params['bad_channels']:
        epoch_post, _ = _interpolate_bad_channels(
            epoch_post, calibration_params['bad_channels'], calibration_params['channel_interpolation_info'])

    # Average reference
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)

    ica = calibration_params['ica']

    # Check ocular ICA components
    source_time_course = ica.get_sources(epoch_post)
    source_data = source_time_course.get_data(copy=True)
    for component_idx in calibration_params['ocular_thresholds_post']:
        component_info = calibration_params['ocular_thresholds_post'][component_idx]
        z_comp = (np.abs(source_data[0, component_idx, component_info['time_indices_of_interest']]) - component_info['mean']) / component_info['std']
        if np.median(z_comp) > trial_reject_opts['ocular']['z_threshold']:
            return False

    # Apply ICA
    ica.apply(epoch_post)

    # Re-baseline and crop
    epoch_post.apply_baseline(cfg.baseline).set_eeg_reference('average', projection=False, verbose=False)
    epoch_post.crop(calibration_params['post_mintime'], None)

    # Apply SOUND
    data = epoch_post.get_data(copy=True)
    data = np.matmul(calibration_params['sound_filter'], data)
    epoch_post = mne.EpochsArray(data, epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0], verbose=False)
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)

    # Apply SSP-SIR
    data = epoch_post.get_data(copy=True)
    data = ssp_sir_single_trial(
        data[0, :, :], calibration_params['sspsir_suppression_matrix_P'],
        calibration_params['sspsir_sir_projmat_suppr'], calibration_params['sspsir_sir_projmat_orig'],
        calibration_params['sspsir_filter_kernel'])
    epoch_post = mne.EpochsArray(data.reshape(1, data.shape[0], data.shape[1]), epoch_post.info,
                                 events=epoch_post.events, tmin=epoch_post.times[0])

    # Second artifact interpolation
    epoch_post = mne.preprocessing.fix_stim_artifact(
        epoch_post, tmin=calibration_params['post_mintime'], tmax=cfg.artifact_window_2[1], mode='window')
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)

    # Global MAD check
    reject_data = epoch_post.copy().crop(cfg.reject_range[0], cfg.reject_range[1]).get_data(copy=True)
    mad_val = median_abs_deviation(reject_data, axis=(1, 2))[0]
    z_mad = (mad_val - calibration_params['good_trial_stats_post']['mads_mean']) / calibration_params['good_trial_stats_post']['mads_std']

    threshold = trial_reject_opts['post']['global_zscore_threshold']
    if z_mad < threshold[0] or z_mad > threshold[1]:
        return False

    # Local MAD check
    local_mad = median_abs_deviation(reject_data, axis=2)
    z_local = zscore(local_mad, axis=1)

    local_threshold = trial_reject_opts['post']['local_zscore_threshold']
    if np.any(np.abs(z_local) > local_threshold):
        return False

    return epoch_post


# ==================== Single-Trial Worker ====================

def process_single_trial(epoch_pre, epoch_post, calibration_params, cfg):
    """Process a single trial (pre, post). Returns (success, epoch_pre, epoch_post)."""
    result_pre = preprocess_pre_trial(epoch_pre, calibration_params, cfg)
    result_post = preprocess_post_trial(epoch_post, calibration_params, cfg)

    if result_pre is False or result_post is False:
        return False, None, None
    return True, result_pre, result_post


_online_trial_worker_state = {}


def _init_online_trial_worker(
    pre_batch_data, pre_batch_info, pre_batch_tmin, pre_batch_events,
    post_batch_data, post_batch_tmin, post_batch_events,
    calibration_params, cfg,
):
    mne.set_log_level("ERROR")
    global _online_trial_worker_state
    _online_trial_worker_state = {
        'pre_batch_data': pre_batch_data,
        'pre_batch_info': pre_batch_info, 
        'pre_batch_tmin': pre_batch_tmin,
        'pre_batch_events': pre_batch_events,
        'post_batch_data': post_batch_data,
        'post_batch_tmin': post_batch_tmin,
        'post_batch_events': post_batch_events,
        'calibration_params': calibration_params,
        'cfg': cfg,
    }


def _process_online_trial_worker(trial_idx):
    s = _online_trial_worker_state
    epoch_pre = mne.EpochsArray(
        s['pre_batch_data'][trial_idx:trial_idx + 1], info=s['pre_batch_info'],
        events=s['pre_batch_events'][trial_idx:trial_idx + 1], tmin=s['pre_batch_tmin'], verbose=False)
    epoch_post = mne.EpochsArray(
        s['post_batch_data'][trial_idx:trial_idx + 1], info=s['pre_batch_info'],
        events=s['post_batch_events'][trial_idx:trial_idx + 1], tmin=s['post_batch_tmin'], verbose=False)
    success, result_pre, result_post = process_single_trial(
        epoch_pre, epoch_post, s['calibration_params'], s['cfg'])
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


def _compute_leadfield():
    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
    forward = mne.read_forward_solution(forward_path)
    return forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0)


def _run_calibration_stage(epochs, cfg, calibration_bundle_path):
    """Calibration only: writes bundle to disk; no state returned except the path."""
    leadfield = _compute_leadfield()

    all_eeg_data = epochs.get_data(copy=False)
    all_events = epochs.events

    print("Appending calibration trials...")

    calibrator = Calibrator(cfg)
    for trial_idx in range(N_TRIALS_CALIBRATE):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        calibrator.add_trial(trial)

    print("Calibrating...")

    start_time = time.time()
    calibration_params, n_successful_trials = calibrator.calibrate(leadfield)

    _save_calibration_bundle(calibration_bundle_path, calibration_params)

    print(f"Used {n_successful_trials} trials for calibration")

    end_time = time.time()
    print(f"Calibration stage took {end_time - start_time:.2f} seconds")


def _process_and_save_trial_group(
    epochs_data, events, info, tmin, cfg, calibration_params, subject_output, subject_id, label
):
    """Batch-process a pre-indexed group of trials and save pre/post epochs to disk."""
    # Batch-crop and resample pre-stim
    epochs_pre_batch = mne.EpochsArray(
        epochs_data, info=info, events=events, tmin=tmin, verbose=False)
    epochs_pre_batch.crop(cfg.pre_range[0], cfg.pre_range[1]).resample(cfg.target_sfreq, method='polyphase')
    pre_batch_data = epochs_pre_batch.get_data(copy=False)
    pre_batch_info = epochs_pre_batch.info
    pre_batch_tmin = epochs_pre_batch.tmin
    pre_batch_events = epochs_pre_batch.events

    # Batch-crop and resample post-stim
    epochs_post_batch = mne.EpochsArray(
        epochs_data, info=info, events=events, tmin=tmin, verbose=False)
    epochs_post_batch.crop(cfg.post_range[0], cfg.post_range[1]).resample(cfg.target_sfreq, method='polyphase')
    post_batch_data = epochs_post_batch.get_data(copy=False)
    post_batch_tmin = epochs_post_batch.tmin
    post_batch_events = epochs_post_batch.events

    del epochs_pre_batch, epochs_post_batch

    n_trials = epochs_data.shape[0]
    pre_list = []
    post_list = []
    bad_trials = []

    initargs = (
        pre_batch_data, pre_batch_info, pre_batch_tmin, pre_batch_events,
        post_batch_data, post_batch_tmin, post_batch_events,
        calibration_params, cfg,
    )
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
):
    """Online trial processing: only config, epochs, and calibration bundle from disk."""
    start_time = time.time()

    calibration_params = _load_calibration_bundle(calibration_bundle_path)
    epochs_data = epochs.get_data(copy=False)

    _process_and_save_trial_group(
        epochs_data[:N_TRIALS_CALIBRATE], epochs.events[:N_TRIALS_CALIBRATE],
        epochs.info, epochs.tmin,
        cfg, calibration_params, subject_output, subject_id, label='calibration',
    )
    _process_and_save_trial_group(
        epochs_data[N_TRIALS_CALIBRATE:], epochs.events[N_TRIALS_CALIBRATE:],
        epochs.info, epochs.tmin,
        cfg, calibration_params, subject_output, subject_id, label='intervention',
    )

    end_time = time.time()
    print(f"Online processing stage took {end_time - start_time:.2f} seconds")


def run_subject_processing(subject_id: str):
    """Main preprocessing pipeline for a single subject."""
    cfg = get_default_config()
    output_path = DATA_ROOT / "processed"
    subject_output = output_path / subject_id
    os.makedirs(subject_output, exist_ok=True)

    calibration_bundle_path = subject_output / f'{subject_id}_calibration_bundle.npy'

    epochs = _load_subject_epochs(subject_id, cfg)
    _run_calibration_stage(epochs, cfg, calibration_bundle_path)

    cfg = get_default_config()
    _run_online_processing_stage(
        epochs, cfg, subject_output, subject_id, calibration_bundle_path)

    print("Done")


# %%
def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for a single subject.")
    parser.add_argument("--subject", required=True, type=str, help="Subject identifier.")
    args = parser.parse_args()
    run_subject_processing(subject_id=args.subject)

if __name__ == "__main__":
    main()
