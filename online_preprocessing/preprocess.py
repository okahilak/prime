#%%
import time

import mne
import numpy as np
from scipy.stats import median_abs_deviation, zscore
from scipy.signal import butter, filtfilt
import warnings
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from utils.ica_calibrator import get_number_of_components, get_ica
from utils.ssp_sir_python import ssp_sir_to_average, ssp_sir_trials, ssp_sir_single_trial
from utils.sound_modified import sound
from utils.channel_interpolations import custom_get_interpolation_matrix, apply_channel_interpolation
from config import get_default_config

DATA_ROOT = Path("~/prime-data").expanduser()

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*ICA\.apply\(\) was baseline-corrected.*",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*Event number greater than 2147483647.*",
)
mne.set_log_level("ERROR")


# ==================== Filtering ====================

def _butter_filter(data, cutoff, btype, fs, order, pad_time):
    nyquist = fs / 2
    if isinstance(cutoff, list):
        if len(cutoff) == 2:
            normalized = [cutoff[0] / nyquist, cutoff[1] / nyquist]
        else:
            raise ValueError("cutoff should be a single value or a list of length 2.")
    else:
        normalized = cutoff / nyquist
    b, a = butter(order, normalized, btype=btype, analog=False)
    filtered = _apply_filter(data, [b, a], pad_time, fs)
    return filtered, [b, a]


def _apply_filter(data, coeffs, pad_time, fs):
    n_pad = int(pad_time * fs)
    padded = np.pad(data, ((0, 0), (0, 0), (n_pad, n_pad)), mode='reflect')
    filtered = filtfilt(coeffs[0], coeffs[1], padded, padlen=None)
    return filtered[:, :, n_pad:-n_pad]


# ==================== Bad Channel Detection ====================

def _get_bad_channels_epoched(epochs, z_mad_threshold, z_power_threshold, freq_range, z_autocorr_threshold, z_auc_threshold):
    """Iteratively detect bad channels using multiple metrics."""
    bad_channels = _get_bad_channels_epoched_once(
        epochs.copy().set_eeg_reference('average', projection=False, verbose=False),
        z_mad_threshold, z_power_threshold, freq_range, z_autocorr_threshold, z_auc_threshold
    )
    if len(bad_channels) > 0:
        while True:
            remaining = epochs.copy().drop_channels(bad_channels).set_eeg_reference('average', projection=False, verbose=False)
            new_bad = _get_bad_channels_epoched_once(remaining, z_mad_threshold, z_power_threshold, freq_range, z_autocorr_threshold, z_auc_threshold)
            if len(new_bad) == 0:
                break
            bad_channels = list(np.union1d(bad_channels, new_bad))
    return bad_channels


def _get_bad_channels_epoched_once(epochs, z_mad_threshold, z_power_threshold, freq_range, z_autocorr_threshold, z_auc_threshold):
    data = epochs.get_data(copy=True)
    n_trials, n_channels, _ = data.shape
    bad_indices = np.array([])

    if z_mad_threshold:
        flattened = data.swapaxes(0, 1).reshape(n_channels, -1)
        mad_vals = median_abs_deviation(flattened, axis=1)
        z_mad = zscore(mad_vals)
        bad_mad = np.where((z_mad > z_mad_threshold[1]) | (z_mad < z_mad_threshold[0]))[0]
        bad_indices = np.union1d(bad_indices, bad_mad)

    if z_power_threshold:
        psds, _ = mne.time_frequency.psd_array_multitaper(data, epochs.info['sfreq'], fmin=freq_range[0], fmax=freq_range[1], verbose=False)
        mean_psd = np.mean(np.mean(psds, axis=0), axis=-1)
        z_psd = zscore(mean_psd)
        bad_power = np.where(z_psd > z_power_threshold)[0]
        bad_indices = np.union1d(bad_indices, bad_power)

    if z_autocorr_threshold:
        autocorrelations = [np.mean([np.corrcoef(data[t, c, :-1], data[t, c, 1:])[0, 1] for t in range(n_trials)]) for c in range(n_channels)]
        z_autocorr = np.abs(zscore(autocorrelations))
        bad_autocorr = np.where(z_autocorr > z_autocorr_threshold)[0]
        bad_indices = np.union1d(bad_indices, bad_autocorr)

    if z_auc_threshold:
        evoked = np.mean(data, axis=0)
        auc_vals = [np.trapz(np.abs(evoked[c, :])) for c in range(n_channels)]
        z_auc = zscore(auc_vals)
        bad_auc = np.where(z_auc > z_auc_threshold)[0]
        bad_indices = np.union1d(bad_indices, bad_auc)

    bad_indices = bad_indices.astype(int)
    return [epochs.ch_names[i] for i in bad_indices]


def _detect_bad_channels_pre(epochs, options, filter_opts):
    data = epochs.get_data(copy=True)
    data, filter_coefficients = _butter_filter(data, filter_opts['cutoff'], filter_opts['btype'], epochs.info['sfreq'], filter_opts['order'], filter_opts['pad_time'])
    filtered_epochs = mne.EpochsArray(data, epochs.info, epochs.events, tmin=epochs.times[0])
    bad_channels = _get_bad_channels_epoched(filtered_epochs, options['z_score_threshold_mad'],
                                        options['z_score_threshold_power'], options['fmin_fmax'],
                                        options['z_score_threshold_autocorr'], False)
    return bad_channels, filter_coefficients


def _detect_bad_channels_post(epochs, options, time_range):
    if time_range:
        cropped = epochs.copy().crop(time_range[0], time_range[1])
    else:
        cropped = epochs.copy()
    return _get_bad_channels_epoched(cropped, False, False, False, False, options['z_score_threshold_auc'])


def _interpolate_bad_channels(epochs, bad_channels, interpolation_info):
    epochs.info['bads'] = bad_channels
    if interpolation_info is None:
        interp_matrix, good_idx, bad_idx = custom_get_interpolation_matrix(epochs, exclude=None, ecog=False)
        interpolation_info = {'interpolation_matrix': interp_matrix, 'goods_idx': good_idx, 'bads_idx': bad_idx}
    apply_channel_interpolation(epochs, interpolation_info)
    epochs.info['bads'] = []
    return epochs, interpolation_info


# ==================== Bad Trial Detection ====================

def _find_bad_trials(epochs, global_z_threshold, local_z_threshold, psd_threshold, psd_freq_range):
    """Iteratively detect bad trials using MAD-based z-scores."""
    bad, stats = _find_bad_trials_once(epochs, global_z_threshold, local_z_threshold, psd_threshold, psd_freq_range, np.array([]))
    if len(bad) > 0:
        while True:
            new_bad, stats = _find_bad_trials_once(epochs, global_z_threshold, local_z_threshold, psd_threshold, psd_freq_range, bad)
            if len(new_bad) == 0:
                break
            bad = np.union1d(bad, new_bad)
    return list(bad.astype(int)), stats


def _find_bad_trials_once(epochs, global_z_threshold, local_z_threshold, psd_threshold, psd_freq_range, known_bad):
    data = epochs.get_data(copy=True)
    n_trials = data.shape[0]
    all_indices = np.arange(n_trials, dtype=int)
    good_indices = np.array([i for i in all_indices if i not in known_bad])
    good_data = data[good_indices, :, :]

    if psd_threshold:
        psds, _ = mne.time_frequency.psd_array_multitaper(good_data, epochs.info['sfreq'], fmin=psd_freq_range[0], fmax=psd_freq_range[1], verbose=False)
        psd_per_trial = np.mean(psds, axis=(1, 2))
        z_psd = zscore(psd_per_trial)
        z_psd_full = []

    mad_per_trial = median_abs_deviation(good_data, axis=(1, 2))
    z_global = zscore(mad_per_trial)

    mad_per_channel = median_abs_deviation(good_data, axis=2)
    z_local = zscore(mad_per_channel, axis=1)

    # Map z-scores back to full trial space
    z_global_full = []
    z_local_full = []
    if psd_threshold:
        z_psd_full = []
    good_idx = 0
    for i in all_indices:
        if i in good_indices:
            z_global_full.append(z_global[good_idx])
            z_local_full.append(z_local[good_idx])
            if psd_threshold:
                z_psd_full.append(z_psd[good_idx])
            good_idx += 1
        else:
            z_global_full.append(0)
            z_local_full.append(np.zeros(z_local.shape[1]))
            if psd_threshold:
                z_psd_full.append(0)

    if len(z_local_full):
        z_local_full = np.vstack(z_local_full)
    else:
        z_local_full = np.zeros((n_trials, mad_per_channel.shape[1]))

    z_global_full = np.asarray(z_global_full)
    if psd_threshold:
        z_psd_full = np.asarray(z_psd_full)

    # Identify bad trials
    bad_local = np.where(np.any(np.abs(z_local_full) > local_z_threshold, axis=1))[0]
    bad_global = np.where((z_global_full > global_z_threshold[1]) | (z_global_full < global_z_threshold[0]))[0]

    # Compute stats from good trials
    bad_global_current = np.where((z_global > global_z_threshold[1]) | (z_global < global_z_threshold[0]))[0]
    good_mad_indices = np.array([i for i in range(len(mad_per_trial)) if i not in bad_global_current])
    stats = {
        'mads': mad_per_trial[good_mad_indices],
        'mads_std': np.std(mad_per_trial[good_mad_indices]),
        'mads_mean': np.mean(mad_per_trial[good_mad_indices]),
    }

    bad = np.union1d(bad_global, bad_local)
    if psd_threshold:
        bad_psd = np.where(np.array(z_psd_full) > psd_threshold)[0]
        bad_psd_current = np.where(z_psd > psd_threshold)[0]
        good_psd_indices = np.array([i for i in range(len(psd_per_trial)) if i not in bad_psd_current])
        stats['psds'] = psd_per_trial[good_psd_indices]
        stats['psds_std'] = np.std(psd_per_trial[good_psd_indices])
        stats['psds_mean'] = np.mean(psd_per_trial[good_psd_indices])
        bad = np.union1d(bad, bad_psd)

    return bad, stats


def _detect_ocular_trials(ica, epochs, tmin, tmax, ocular_component_indices, z_threshold):
    """Detect trials with ocular artifacts from ICA source time courses."""
    sources = ica.get_sources(epochs)
    source_data = sources.get_data(copy=True)

    if tmax is None:
        tmax = np.inf
    if tmin is None:
        tmin = -np.inf

    time_mask = np.where((sources.times >= tmin) & (sources.times <= tmax))[0]
    bad_trials = np.array([])
    thresholds = {}

    for component_idx in ocular_component_indices:
        abs_time_course = np.abs(source_data[:, component_idx, :])
        n_trials, n_times = abs_time_course.shape
        flat_time_course = abs_time_course.ravel()
        z_scored = zscore(flat_time_course).reshape(n_trials, n_times)

        assert z_scored.shape == (n_trials, n_times)

        median_z = np.median(z_scored[:, time_mask], axis=1)
        bad_now = np.where(median_z > z_threshold)[0]
        thresholds[component_idx] = {
            'std': np.std(flat_time_course),
            'mean': np.mean(flat_time_course),
            'time_indices_of_interest': time_mask,
        }
        bad_trials = np.union1d(bad_trials, bad_now)

    return list(bad_trials.astype(int)), thresholds


def _drop_bad_trials(epoch_list, bad_indices):
    for epoch in epoch_list:
        epoch.drop(bad_indices)
    return epoch_list


# ==================== Main Calibration Pipeline ====================

def _single_trial_epochs_from_arrays(eeg_data, events, epochs, trial_idx):
    """One trial as EpochsArray without deep-copying the full subject epochs."""
    return mne.EpochsArray(
        eeg_data[trial_idx:trial_idx + 1],
        info=epochs.info,
        events=events[trial_idx:trial_idx + 1],
        tmin=epochs.tmin,
        verbose=False,
    )


def append_calibration_trial(epochs_pre, epochs_pre_ica, epochs_post, trial, cfg, ica_time_range):
    """Append one raw trial (cropped and resampled) to calibration epoch structs."""
    trial_pre = trial.copy().crop(cfg.pre_range[0], cfg.pre_range[1])
    trial_pre_ica = trial.copy().crop(ica_time_range[0], ica_time_range[1])
    trial_post = trial.copy().crop(cfg.post_range[0], cfg.post_range[1])
    for segment in (trial_pre, trial_pre_ica, trial_post):
        segment.resample(cfg.target_sfreq, method='polyphase')
    if epochs_pre is None:
        return trial_pre, trial_pre_ica, trial_post
    epochs_pre = mne.concatenate_epochs([epochs_pre, trial_pre])
    epochs_pre_ica = mne.concatenate_epochs([epochs_pre_ica, trial_pre_ica])
    epochs_post = mne.concatenate_epochs([epochs_post, trial_post])
    return epochs_pre, epochs_pre_ica, epochs_post


def preprocess_calibration(epochs_pre, epochs_pre_ica, epochs_post, cfg, opts, leadfield):
    """Full calibration preprocessing pipeline for pre-stim and post-stim epochs."""
    channel_reject_opts = opts['channel_reject_opts']
    ica_opts = opts['ica_opts']
    trial_reject_opts = opts['trial_reject_opts']
    sound_opts = opts['sound_opts']
    ssp_sir_opts = opts['ssp_sir_opts']
    filter_opts = opts['filter_opts']

    calibration_params = {}

    # Post-stim: baseline and artifact interpolation
    epochs_post.apply_baseline(cfg.baseline)
    epochs_post = mne.preprocessing.fix_stim_artifact(
        epochs_post, tmin=cfg.artifact_window_1[0], tmax=cfg.artifact_window_1[1], mode='window')

    # Detect and interpolate bad channels
    bad_channels_pre, pre_filter_coefficients = _detect_bad_channels_pre(
        epochs_pre, channel_reject_opts['pre'], filter_opts)
    calibration_params['pre_stim_filter'] = pre_filter_coefficients
    bad_channels_post = _detect_bad_channels_post(
        epochs_post, channel_reject_opts['post'], cfg.reject_range)
    bad_channels = list(np.union1d(bad_channels_pre, bad_channels_post))

    if bad_channels:
        epochs_pre, interpolation_info = _interpolate_bad_channels(epochs_pre, bad_channels, None)
        epochs_pre_ica, _ = _interpolate_bad_channels(epochs_pre_ica, bad_channels, interpolation_info)
        epochs_post, _ = _interpolate_bad_channels(epochs_post, bad_channels, interpolation_info)
    else:
        interpolation_info = None
    calibration_params['bad_channels'] = bad_channels
    calibration_params['channel_interpolation_info'] = interpolation_info

    # ICA calibration on filtered pre-stim
    ica_data = epochs_pre_ica.get_data(copy=True)
    ica_data, _ = _butter_filter(ica_data, ica_opts['filtering']['cutoff'], 'bandpass',
                                               epochs_pre_ica.info['sfreq'], ica_opts['filtering']['order_bandpass'],
                                               ica_opts['filtering']['pad_time_bandpass'])
    epochs_pre_ica_filtered = mne.EpochsArray(ica_data, info=epochs_pre_ica.info, events=epochs_pre_ica.events, tmin=epochs_pre_ica.times[0])

    epochs_post.set_eeg_reference('average', projection=False, verbose=False)
    epochs_pre_ica_filtered.set_eeg_reference('average', projection=False, verbose=False)

    n_components = get_number_of_components(epochs_pre_ica_filtered.get_data(copy=True), ica_opts['pc_threshold'])
    ica, excluded_components, _ = get_ica(epochs_pre_ica_filtered, n_components, None,
                                             ica_opts['bad_component_thresholds'],
                                             ica_opts['n_min_comps_to_reject'],
                                             ica_opts['threshold_min_components_to_reject'])
    del epochs_pre_ica_filtered

    # Filter pre-stim
    pre_data = epochs_pre.get_data(copy=True)
    pre_data = _apply_filter(pre_data, pre_filter_coefficients, filter_opts['pad_time'], epochs_pre.info['sfreq'])
    epochs_pre = mne.EpochsArray(pre_data, info=epochs_pre.info, events=epochs_pre.events, tmin=epochs_pre.times[0])
    if cfg.use_ica_on_pre:
        epochs_pre.set_eeg_reference('average', projection=False, verbose=False)
    del pre_data

    # Detect ocular artifact trials
    if cfg.use_ica_on_pre:
        bad_ocular_pre, _ = _detect_ocular_trials(
            ica, epochs_pre, trial_reject_opts['ocular']['pre_timerange_min'], None,
            excluded_components['eye blink'], trial_reject_opts['ocular']['z_threshold'])

    bad_ocular_post, ocular_threshold_post = _detect_ocular_trials(
        ica, epochs_post, trial_reject_opts['ocular']['post_timerange'][0],
        trial_reject_opts['ocular']['post_timerange'][1],
        excluded_components['eye blink'], trial_reject_opts['ocular']['z_threshold'])

    if cfg.use_ica_on_pre:
        bad_ocular = list(np.union1d(bad_ocular_pre, bad_ocular_post).astype(int))
    else:
        bad_ocular = bad_ocular_post

    calibration_params['ocular_thresholds_post'] = ocular_threshold_post

    # Apply ICA
    if cfg.use_ica_on_pre:
        ica.apply(epochs_pre)
    ica.apply(epochs_post)

    # Drop ocular-bad trials
    epochs_pre, epochs_post = _drop_bad_trials([epochs_pre, epochs_post], bad_ocular)
    epochs_pre.set_eeg_reference('average', projection=False, verbose=False)

    # Detect bad pre-stim trials
    bad_pre, stats_pre = _find_bad_trials(epochs_pre, trial_reject_opts['pre']['global_zscore_threshold'],
                                          trial_reject_opts['pre']['local_zscore_threshold'], False, False)
    calibration_params['good_trial_stats_pre'] = stats_pre
    epochs_pre, epochs_post = _drop_bad_trials([epochs_pre, epochs_post], bad_pre)

    # Apply SOUND to post-stim
    epochs_post.apply_baseline(cfg.baseline).set_eeg_reference('average', projection=False, verbose=False)
    post_t0 = epochs_post.times[epochs_post.times > 0][0]
    calibration_params['post_mintime'] = post_t0
    epochs_post.crop(post_t0, None)

    post_data = epochs_post.get_data(copy=True)
    evoked = np.mean(post_data, axis=0)
    n_channels = evoked.shape[0]
    sound_filter, _, _, _ = sound(
        evoked.T, 0, np.ones((n_channels, 1)), n_channels, leadfield,
        sound_opts['max_iterations'], sound_opts['lambda'],
        sound_opts['convergence_tolerance'], sound_opts['fixed_max_iterations'])

    for i in range(post_data.shape[0]):
        post_data[i, :, :] = np.matmul(sound_filter, post_data[i, :, :])

    epochs_post = mne.EpochsArray(post_data, epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0])
    epochs_post.set_eeg_reference('average', projection=False, verbose=False)
    calibration_params['sound_filter'] = sound_filter

    # Apply SSP-SIR to post-stim
    post_data = epochs_post.get_data(copy=True)
    evoked = np.mean(post_data, axis=0)
    _, _, _, kernel, P, _, _, projection_suppression, projection_original, _ = ssp_sir_to_average(
        evoked, leadfield, epochs_post.info['sfreq'], ssp_sir_opts['timerange'], method=ssp_sir_opts['method'])

    post_data = ssp_sir_trials(post_data, P, projection_suppression, projection_original, kernel)
    epochs_post = mne.EpochsArray(post_data, epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0])

    calibration_params['sspsir_suppression_matrix_P'] = P
    calibration_params['sspsir_filter_kernel'] = kernel
    calibration_params['sspsir_sir_projmat_suppr'] = projection_suppression
    calibration_params['sspsir_sir_projmat_orig'] = projection_original

    # Second artifact interpolation
    epochs_post = mne.preprocessing.fix_stim_artifact(
        epochs_post, tmin=post_t0, tmax=cfg.artifact_window_2[1], mode='window')
    epochs_post.set_eeg_reference('average', projection=False, verbose=False)

    # Detect bad post-stim trials
    bad_post, stats_post = _find_bad_trials(
        epochs_post.copy().crop(cfg.reject_range[0], cfg.reject_range[1]),
        trial_reject_opts['post']['global_zscore_threshold'],
        trial_reject_opts['post']['local_zscore_threshold'], False, False)
    calibration_params['good_trial_stats_post'] = stats_post
    epochs_pre, epochs_post = _drop_bad_trials([epochs_pre, epochs_post], bad_post)

    n_successful_trials = epochs_pre.get_data(copy=True).shape[0]

    calibration_params['ica'] = ica
    return calibration_params, n_successful_trials


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
    opts = cfg.to_dicts()
    leadfield = _compute_leadfield()
    ica_time_range = opts['ica_opts']['pre_timerange']

    all_eeg_data = epochs.get_data(copy=False)
    all_events = epochs.events

    print("Appending calibration trials...")

    epochs_pre = epochs_pre_ica = epochs_post = None
    for trial_idx in range(cfg.n_trials_calibrate):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        epochs_pre, epochs_pre_ica, epochs_post = append_calibration_trial(
            epochs_pre, epochs_pre_ica, epochs_post, trial, cfg, ica_time_range)

    print("Calibrating...")

    start_time = time.time()
    calibration_params, n_successful_trials = preprocess_calibration(
        epochs_pre.copy(), epochs_pre_ica.copy(), epochs_post.copy(), cfg, opts, leadfield)

    _save_calibration_bundle(calibration_bundle_path, calibration_params)

    print(f"Used {n_successful_trials} trials for calibration")

    end_time = time.time()
    print(f"Calibration stage took {end_time - start_time:.2f} seconds")


def _run_online_processing_stage(
    epochs, cfg, subject_output, subject_id, calibration_bundle_path,
):
    """Online trial processing: only config, epochs, and calibration bundle from disk."""
    start_time = time.time()

    calibration_params = _load_calibration_bundle(calibration_bundle_path)

    n_total = len(epochs)
    all_eeg_data = epochs.get_data(copy=False)

    # Batch-crop and resample pre-stim epochs for all trials
    epochs_pre_batch = mne.EpochsArray(
        all_eeg_data, info=epochs.info, events=epochs.events, tmin=epochs.tmin, verbose=False)
    epochs_pre_batch.crop(cfg.pre_range[0], cfg.pre_range[1]).resample(cfg.target_sfreq, method='polyphase')
    pre_batch_data = epochs_pre_batch.get_data(copy=False)
    pre_batch_info = epochs_pre_batch.info
    pre_batch_tmin = epochs_pre_batch.tmin
    pre_batch_events = epochs_pre_batch.events

    # Batch-crop and resample post-stim epochs for all trials
    epochs_post_batch = mne.EpochsArray(
        all_eeg_data, info=epochs.info, events=epochs.events, tmin=epochs.tmin, verbose=False)
    epochs_post_batch.crop(cfg.post_range[0], cfg.post_range[1]).resample(cfg.target_sfreq, method='polyphase')
    post_batch_data = epochs_post_batch.get_data(copy=False)
    post_batch_tmin = epochs_post_batch.tmin
    post_batch_events = epochs_post_batch.events

    # Free memory from originals
    del epochs_pre_batch, epochs_post_batch

    # --- Parallel single-trial processing ---
    pre_list = []
    post_list = []
    bad_trials_online = []

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
        results = list(executor.map(_process_online_trial_worker, range(n_total)))

    for i, (_, success, result_pre, result_post) in enumerate(results):
        if success:
            pre_list.append(result_pre)
            post_list.append(result_post)
        else:
            bad_trials_online.append(i)

    epochs_pre_final = mne.concatenate_epochs(pre_list)
    epochs_post_final = mne.concatenate_epochs(post_list)

    # --- Save results ---
    for epoch, label in zip([epochs_pre_final, epochs_post_final], ['pre', 'post']):
        epoch.save(os.path.join(subject_output, f"{subject_id}_{label}.fif"), overwrite=True)

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
