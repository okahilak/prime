#%%
import mne
import numpy as np
import scipy
import scipy.io
from scipy.stats import median_abs_deviation, zscore
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
import warnings
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ica_calibrator import get_number_of_components, get_ica
from ssp_sir_python import ssp_sir_to_average, ssp_sir_trials, ssp_sir_single_trial
from sound_modified import sound
from channel_interpolations import custom_get_interpolation_matrix, apply_channel_interpolation
from config import get_default_config

_PREP_ROOT = Path(__file__).resolve().parents[2]

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*ICA\.apply\(\) was baseline-corrected.*"
)


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

def _get_bad_channels_epoched(epochs, z_mad_thresh, z_power_thresh, freq_range, z_autocorr_thresh, z_auc_thresh):
    """Iteratively detect bad channels using multiple metrics."""
    bad_channels = _get_bad_channels_epoched_once(
        epochs.copy().set_eeg_reference('average', projection=False, verbose=False),
        z_mad_thresh, z_power_thresh, freq_range, z_autocorr_thresh, z_auc_thresh
    )
    if len(bad_channels) > 0:
        while True:
            remaining = epochs.copy().drop_channels(bad_channels).set_eeg_reference('average', projection=False, verbose=False)
            new_bad = _get_bad_channels_epoched_once(remaining, z_mad_thresh, z_power_thresh, freq_range, z_autocorr_thresh, z_auc_thresh)
            if len(new_bad) == 0:
                break
            bad_channels = list(np.union1d(bad_channels, new_bad))
    return bad_channels


def _get_bad_channels_epoched_once(epochs, z_mad_thresh, z_power_thresh, freq_range, z_autocorr_thresh, z_auc_thresh):
    data = epochs.get_data(copy=True)
    n_trials, n_channels, _ = data.shape
    bad_indices = np.array([])

    if z_mad_thresh:
        flattened = data.swapaxes(0, 1).reshape(n_channels, -1)
        mad_vals = median_abs_deviation(flattened, axis=1)
        z_mad = zscore(mad_vals)
        bad_mad = np.where((z_mad > z_mad_thresh[1]) | (z_mad < z_mad_thresh[0]))[0]
        bad_indices = np.union1d(bad_indices, bad_mad)

    if z_power_thresh:
        psds, _ = mne.time_frequency.psd_array_multitaper(data, epochs.info['sfreq'], fmin=freq_range[0], fmax=freq_range[1], verbose=False)
        mean_psd = np.mean(np.mean(psds, axis=0), axis=-1)
        z_psd = zscore(mean_psd)
        bad_power = np.where(z_psd > z_power_thresh)[0]
        bad_indices = np.union1d(bad_indices, bad_power)

    if z_autocorr_thresh:
        autocorrelations = [np.mean([np.corrcoef(data[t, c, :-1], data[t, c, 1:])[0, 1] for t in range(n_trials)]) for c in range(n_channels)]
        z_autocorr = np.abs(zscore(autocorrelations))
        bad_autocorr = np.where(z_autocorr > z_autocorr_thresh)[0]
        bad_indices = np.union1d(bad_indices, bad_autocorr)

    if z_auc_thresh:
        evoked = np.mean(data, axis=0)
        auc_vals = [np.trapz(np.abs(evoked[c, :])) for c in range(n_channels)]
        z_auc = zscore(auc_vals)
        bad_auc = np.where(z_auc > z_auc_thresh)[0]
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

def _find_bad_trials(epochs, global_z_thresh, local_z_thresh, psd_thresh, psd_freq_range):
    """Iteratively detect bad trials using MAD-based z-scores."""
    bad, stats = _find_bad_trials_once(epochs, global_z_thresh, local_z_thresh, psd_thresh, psd_freq_range, np.array([]))
    if len(bad) > 0:
        while True:
            new_bad, stats = _find_bad_trials_once(epochs, global_z_thresh, local_z_thresh, psd_thresh, psd_freq_range, bad)
            if len(new_bad) == 0:
                break
            bad = np.union1d(bad, new_bad)
    return list(bad.astype(int)), stats


def _find_bad_trials_once(epochs, global_z_thresh, local_z_thresh, psd_thresh, psd_freq_range, known_bad):
    data = epochs.get_data(copy=True)
    n_trials = data.shape[0]
    all_indices = np.arange(n_trials, dtype=int)
    good_indices = np.array([i for i in all_indices if i not in known_bad])
    good_data = data[good_indices, :, :]

    if psd_thresh:
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
    if psd_thresh:
        z_psd_full = []
    good_idx = 0
    for i in all_indices:
        if i in good_indices:
            z_global_full.append(z_global[good_idx])
            z_local_full.append(z_local[good_idx])
            if psd_thresh:
                z_psd_full.append(z_psd[good_idx])
            good_idx += 1
        else:
            z_global_full.append(0)
            z_local_full.append(np.zeros(z_local.shape[1]))
            if psd_thresh:
                z_psd_full.append(0)

    if len(z_local_full):
        z_local_full = np.vstack(z_local_full)
    else:
        z_local_full = np.zeros((n_trials, mad_per_channel.shape[1]))

    z_global_full = np.asarray(z_global_full)
    if psd_thresh:
        z_psd_full = np.asarray(z_psd_full)

    # Identify bad trials
    bad_local = np.where(np.any(np.abs(z_local_full) > local_z_thresh, axis=1))[0]
    bad_global = np.where((z_global_full > global_z_thresh[1]) | (z_global_full < global_z_thresh[0]))[0]

    # Compute stats from good trials
    bad_global_current = np.where((z_global > global_z_thresh[1]) | (z_global < global_z_thresh[0]))[0]
    good_mad_indices = np.array([i for i in range(len(mad_per_trial)) if i not in bad_global_current])
    stats = {
        'mads': mad_per_trial[good_mad_indices],
        'mads_std': np.std(mad_per_trial[good_mad_indices]),
        'mads_mean': np.mean(mad_per_trial[good_mad_indices]),
    }

    bad = np.union1d(bad_global, bad_local)
    if psd_thresh:
        bad_psd = np.where(np.array(z_psd_full) > psd_thresh)[0]
        bad_psd_current = np.where(z_psd > psd_thresh)[0]
        good_psd_indices = np.array([i for i in range(len(psd_per_trial)) if i not in bad_psd_current])
        stats['psds'] = psd_per_trial[good_psd_indices]
        stats['psds_std'] = np.std(psd_per_trial[good_psd_indices])
        stats['psds_mean'] = np.mean(psd_per_trial[good_psd_indices])
        bad = np.union1d(bad, bad_psd)

    return bad, stats


def _detect_ocular_trials(ica, epochs, tmin, tmax, ocular_component_indices, z_thresh):
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
        bad_now = np.where(median_z > z_thresh)[0]
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


# ==================== EMG Processing ====================

def _detect_bad_emg_trials(epochs_emg, pre_innervation_opts, ptp_opts, line_freq):
    """Check EMG trials for pre-innervation and valid MEP."""
    times = epochs_emg.times
    data = epochs_emg.get_data(copy=True)
    pre_innervation_mask = np.where((times >= pre_innervation_opts['tmin']) & (times <= pre_innervation_opts['tmax']))[0]
    ptp_mask = np.where((times >= ptp_opts['tmin']) & (times <= ptp_opts['tmax']))[0]
    pre_stim_times = times[pre_innervation_mask]
    n_channels = data.shape[1]
    n_trials = data.shape[0]

    time_info = {
        'full_emg_times': times,
        'pre_innervation_time_indices': pre_innervation_mask,
        'peak_to_peak_time_indices': ptp_mask,
        'pre_stim_times': pre_stim_times,
    }

    bad_trials = []

    for channel_idx in range(n_channels):
        for trial_idx in range(n_trials):
            has_pre_innervation, valid_ptp, _, data = _prep_emg_trial(
                data, trial_idx, channel_idx, pre_innervation_mask, line_freq,
                pre_stim_times, times, pre_innervation_opts, ptp_mask, ptp_opts,
                epochs_emg.info['sfreq']
            )
            if has_pre_innervation or not valid_ptp:
                bad_trials.append(trial_idx)

    bad_trials = np.unique(bad_trials)
    epochs_emg = mne.EpochsArray(data, info=epochs_emg.info, events=epochs_emg.events, tmin=epochs_emg.times[0])
    return bad_trials, epochs_emg, time_info


def _prep_emg_trial(data, trial_idx, channel_idx, pre_innervation_mask, line_freq,
                    pre_stim_times, full_times, pre_innervation_opts, ptp_mask, ptp_opts, sfreq):
    """Remove line-frequency sine, check pre-innervation and MEP validity for one trial."""
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=scipy.optimize.OptimizeWarning)
        for harmonic in [1, 2]:
            try:
                pre_data = data[trial_idx, channel_idx, pre_innervation_mask]
                sine_model = _make_sine_model(line_freq * harmonic)
                sine_fit = _fit_line_freq_sine(pre_data, pre_stim_times, full_times, sine_model)
                data[trial_idx, channel_idx, :] -= sine_fit
            except scipy.optimize.OptimizeWarning:
                break

    emg_pre = data[trial_idx, channel_idx, pre_innervation_mask]
    has_pre_innervation = (np.max(emg_pre) - np.min(emg_pre)) > pre_innervation_opts['threshold']

    emg_ptp = data[trial_idx, channel_idx, ptp_mask]
    if ptp_opts['check_ptp']:
        min_dist_samples = ptp_opts['min_distance'] * sfreq
        peaks_pos, _ = find_peaks(emg_ptp, prominence=ptp_opts['prominence'], distance=min_dist_samples)
        peaks_neg, _ = find_peaks(-emg_ptp, prominence=ptp_opts['prominence'], distance=min_dist_samples)
        if len(peaks_pos) < 1 or len(peaks_neg) < 1:
            return has_pre_innervation, False, False, data
        ptp = np.max(emg_ptp[peaks_pos]) - np.min(emg_ptp[peaks_neg])
        valid_ptp = ptp >= ptp_opts['min_ptp_height']
    else:
        ptp = np.max(emg_ptp) - np.min(emg_ptp)
        valid_ptp = True

    return has_pre_innervation, valid_ptp, ptp, data


def _fit_line_freq_sine(pre_data, pre_times, full_times, sine_model):
    p0 = [np.std(pre_data) * np.sqrt(2), 0, np.mean(pre_data)]
    popt, _ = curve_fit(sine_model, pre_times, pre_data, p0=p0)
    return sine_model(full_times, *popt)


def _make_sine_model(freq):
    def model(t, amplitude, phase, offset):
        return amplitude * np.sin(2 * np.pi * freq * t + phase) + offset
    return model


def _select_best_emg_channel(epochs_emg, ptp_opts, filter_opts):
    data = epochs_emg.get_data(copy=True)
    data, filter_coefficients = _butter_filter(data, filter_opts['cutoff'], filter_opts['btype'],
                                       epochs_emg.info['sfreq'], filter_opts['order'], filter_opts['pad_time'])
    epochs_emg = mne.EpochsArray(data, info=epochs_emg.info, events=epochs_emg.events, tmin=epochs_emg.times[0])
    cropped = epochs_emg.copy().crop(ptp_opts['tmin'], ptp_opts['tmax']).get_data(copy=True)
    ptp_per_channel = np.mean(np.max(cropped, axis=2) - np.min(cropped, axis=2), axis=0)
    best_idx = np.argmax(ptp_per_channel)
    return epochs_emg.ch_names[best_idx], filter_coefficients


# ==================== Main Calibration Pipeline ====================

def preprocess_calibration(epochs, epochs_emg, pre_range, post_range, baseline,
                           artifact_window_1, artifact_window_2, reject_range,
                           emg_time_range, channel_reject_opts, ica_opts, trial_reject_opts,
                           emg_reject_opts, sound_opts, ssp_sir_opts, leadfield,
                           filter_opts, filter_opts_emg, line_freq, target_sfreq,
                           n_trials_goal, use_ica_on_pre, emg_filter_coefficients):
    """Full calibration preprocessing pipeline for pre-stim, post-stim, and EMG epochs."""
    calibration_params = {}
    rejected_trials = {}
    ica_time_range = ica_opts['pre_timerange']

    # Crop into segments
    epochs_emg.crop(emg_time_range[0], emg_time_range[1])
    epochs_pre = epochs.copy().crop(pre_range[0], pre_range[1])
    epochs_pre_ica = epochs.copy().crop(ica_time_range[0], ica_time_range[1])
    epochs_post = epochs.copy().crop(post_range[0], post_range[1])
    del epochs

    # Resample
    for epoch in [epochs_pre, epochs_pre_ica, epochs_post, epochs_emg]:
        epoch.resample(target_sfreq, method='polyphase')

    # Post-stim: baseline and artifact interpolation
    epochs_post.apply_baseline(baseline)
    epochs_post = mne.preprocessing.fix_stim_artifact(epochs_post, tmin=artifact_window_1[0], tmax=artifact_window_1[1], mode='window')

    # Detect and interpolate bad channels
    bad_channels_pre, pre_filter_coefficients = _detect_bad_channels_pre(epochs_pre, channel_reject_opts['pre'], filter_opts)
    calibration_params['pre_stim_filter'] = pre_filter_coefficients
    bad_channels_post = _detect_bad_channels_post(epochs_post, channel_reject_opts['post'], reject_range)
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
    if use_ica_on_pre:
        epochs_pre.set_eeg_reference('average', projection=False, verbose=False)
    del pre_data

    # Detect ocular artifact trials
    if use_ica_on_pre:
        bad_ocular_pre, _ = _detect_ocular_trials(
            ica, epochs_pre, trial_reject_opts['ocular']['pre_timerange_min'], None,
            excluded_components['eye blink'], trial_reject_opts['ocular']['z_thresh'])

    bad_ocular_post, ocular_threshold_post = _detect_ocular_trials(
        ica, epochs_post, trial_reject_opts['ocular']['post_timerange'][0],
        trial_reject_opts['ocular']['post_timerange'][1],
        excluded_components['eye blink'], trial_reject_opts['ocular']['z_thresh'])

    if use_ica_on_pre:
        bad_ocular = list(np.union1d(bad_ocular_pre, bad_ocular_post).astype(int))
    else:
        bad_ocular = bad_ocular_post

    rejected_trials['bad_trials_ocular'] = bad_ocular
    calibration_params['ocular_thresholds_post'] = ocular_threshold_post

    # Apply ICA
    if use_ica_on_pre:
        ica.apply(epochs_pre)
    ica.apply(epochs_post)

    # Drop ocular-bad trials
    epochs_pre, epochs_post, epochs_emg = _drop_bad_trials([epochs_pre, epochs_post, epochs_emg], bad_ocular)
    epochs_pre.set_eeg_reference('average', projection=False, verbose=False)

    # Detect bad pre-stim trials
    bad_pre, stats_pre = _find_bad_trials(epochs_pre, trial_reject_opts['pre']['global_zscore_threshold'],
                                          trial_reject_opts['pre']['local_zscore_threshold'], False, False)
    rejected_trials['bad_trials_pre'] = bad_pre
    calibration_params['good_trial_stats_pre'] = stats_pre
    epochs_pre, epochs_post, epochs_emg = _drop_bad_trials([epochs_pre, epochs_post, epochs_emg], bad_pre)

    # Apply SOUND to post-stim
    epochs_post.apply_baseline(baseline).set_eeg_reference('average', projection=False, verbose=False)
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
    epochs_post = mne.preprocessing.fix_stim_artifact(epochs_post, tmin=post_t0, tmax=artifact_window_2[1], mode='window')
    epochs_post.set_eeg_reference('average', projection=False, verbose=False)

    # Detect bad post-stim trials
    bad_post, stats_post = _find_bad_trials(
        epochs_post.copy().crop(reject_range[0], reject_range[1]),
        trial_reject_opts['post']['global_zscore_threshold'],
        trial_reject_opts['post']['local_zscore_threshold'], False, False)
    rejected_trials['bad_trials_post'] = bad_post
    calibration_params['good_trial_stats_post'] = stats_post
    epochs_pre, epochs_post, epochs_emg = _drop_bad_trials([epochs_pre, epochs_post, epochs_emg], bad_post)

    # EMG processing
    emg_data = epochs_emg.get_data(copy=True)
    emg_data = _apply_filter(emg_data, emg_filter_coefficients, filter_opts_emg['pad_time'], epochs_emg.info['sfreq'])
    epochs_emg = mne.EpochsArray(emg_data, info=epochs_emg.info, events=epochs_emg.events, tmin=epochs_emg.times[0])
    bad_emg, epochs_emg, emg_time_info = _detect_bad_emg_trials(
        epochs_emg, emg_reject_opts['pre_innervation_options'], emg_reject_opts['ptp_options'], line_freq)

    calibration_params['emg_filter'] = emg_filter_coefficients
    calibration_params['emg_prep_times'] = emg_time_info
    rejected_trials['bad_trials_emg'] = bad_emg

    epochs_pre, epochs_post, epochs_emg = _drop_bad_trials([epochs_pre, epochs_post, epochs_emg], bad_emg)
    n_trials_left = epochs_pre.get_data(copy=True).shape[0]

    if n_trials_left < n_trials_goal:
        return n_trials_goal - n_trials_left

    return epochs_pre, epochs_post, ica, calibration_params, rejected_trials


# ==================== Single-Trial Processing ====================

def preprocess_pre_trial(epoch_pre, info, trial_reject_opts, filter_opts, use_ica_on_pre):
    """Preprocess a single pre-stimulus trial using calibrated parameters."""
    # Interpolate bad channels
    if info['bad_channels']:
        epoch_pre, _ = _interpolate_bad_channels(epoch_pre, info['bad_channels'], info['channel_interpolation_info'])

    if use_ica_on_pre:
        raise NotImplementedError("Applying ICA to pre-stimulus in single-trials is not currently implemented")

    data = epoch_pre.get_data(copy=True)

    # Apply filter
    data = _apply_filter(data, info['pre_stim_filter'], filter_opts['pad_time'], epoch_pre.info['sfreq'])

    # Mean subtraction (average reference)
    data -= np.mean(data, axis=1)

    # Global MAD check
    mad_val = median_abs_deviation(data, axis=(1, 2))[0]
    z_mad = (mad_val - info['good_trial_stats_pre']['mads_mean']) / info['good_trial_stats_pre']['mads_std']

    thresh = trial_reject_opts['pre']['global_zscore_threshold']
    if z_mad < thresh[0] or z_mad > thresh[1]:
        return False

    # Local MAD check
    local_mad = median_abs_deviation(data, axis=2)
    z_local = zscore(local_mad, axis=1)

    local_thresh = trial_reject_opts['pre']['local_zscore_threshold']
    if np.any(np.abs(z_local) > local_thresh):
        return False

    # Rebuild epoch
    epoch_pre = mne.EpochsArray(data, info=epoch_pre.info, events=epoch_pre.events, tmin=epoch_pre.times[0], verbose=False)
    return epoch_pre


def preprocess_post_trial(epoch_post, info, trial_reject_opts, ica, baseline, artifact_window_1, artifact_window_2, leadfield, reject_range):
    """Preprocess a single post-stimulus trial using calibrated parameters."""
    # Baseline correction
    epoch_post.apply_baseline(baseline)

    # First artifact interpolation
    epoch_post = mne.preprocessing.fix_stim_artifact(epoch_post, tmin=artifact_window_1[0], tmax=artifact_window_1[1], mode='window')

    # Interpolate bad channels
    if info['bad_channels']:
        epoch_post, _ = _interpolate_bad_channels(epoch_post, info['bad_channels'], info['channel_interpolation_info'])

    # Average reference
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)

    # Check ocular ICA components
    source_time_course = ica.get_sources(epoch_post)
    source_data = source_time_course.get_data(copy=True)
    for component_idx in info['ocular_thresholds_post']:
        component_info = info['ocular_thresholds_post'][component_idx]
        z_comp = (np.abs(source_data[0, component_idx, component_info['time_indices_of_interest']]) - component_info['mean']) / component_info['std']
        if np.median(z_comp) > trial_reject_opts['ocular']['z_thresh']:
            return False

    # Apply ICA
    ica.apply(epoch_post)

    # Re-baseline and crop
    epoch_post.apply_baseline(baseline).set_eeg_reference('average', projection=False, verbose=False)
    epoch_post.crop(info['post_mintime'], None)

    # Apply SOUND
    data = epoch_post.get_data(copy=True)
    data = np.matmul(info['sound_filter'], data)
    epoch_post = mne.EpochsArray(data, epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0], verbose=False)
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)

    # Apply SSP-SIR
    data = epoch_post.get_data(copy=True)
    data = ssp_sir_single_trial(data[0, :, :], info['sspsir_suppression_matrix_P'],
                                info['sspsir_sir_projmat_suppr'], info['sspsir_sir_projmat_orig'],
                                info['sspsir_filter_kernel'])
    epoch_post = mne.EpochsArray(data.reshape(1, data.shape[0], data.shape[1]), epoch_post.info,
                                 events=epoch_post.events, tmin=epoch_post.times[0])

    # Second artifact interpolation
    epoch_post = mne.preprocessing.fix_stim_artifact(epoch_post, tmin=info['post_mintime'], tmax=artifact_window_2[1], mode='window')
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)

    # Global MAD check
    reject_data = epoch_post.copy().crop(reject_range[0], reject_range[1]).get_data(copy=True)
    mad_val = median_abs_deviation(reject_data, axis=(1, 2))[0]
    z_mad = (mad_val - info['good_trial_stats_post']['mads_mean']) / info['good_trial_stats_post']['mads_std']

    thresh = trial_reject_opts['post']['global_zscore_threshold']
    if z_mad < thresh[0] or z_mad > thresh[1]:
        return False

    # Local MAD check
    local_mad = median_abs_deviation(reject_data, axis=2)
    z_local = zscore(local_mad, axis=1)

    local_thresh = trial_reject_opts['post']['local_zscore_threshold']
    if np.any(np.abs(z_local) > local_thresh):
        return False

    return epoch_post


def preprocess_emg_trial(epoch_emg, filter_opts_emg, emg_reject_opts, info, line_freq):
    """Preprocess a single EMG trial and determine MEP validity."""
    full_times = info['emg_prep_times']['full_emg_times']
    pre_stim_times = full_times[info['emg_prep_times']['pre_innervation_time_indices']]
    data = epoch_emg.get_data(copy=True)

    # Apply filter
    data = _apply_filter(data, info['emg_filter'], filter_opts_emg['pad_time'], epoch_emg.info['sfreq'])

    # Check pre-innervation and MEP
    has_pre_innervation, valid_ptp, _, data = _prep_emg_trial(
        data, 0, 0,
        info['emg_prep_times']['pre_innervation_time_indices'],
        line_freq, pre_stim_times, full_times,
        emg_reject_opts['pre_innervation_options'],
        info['emg_prep_times']['peak_to_peak_time_indices'],
        emg_reject_opts['ptp_options'],
        epoch_emg.info['sfreq']
    )

    if has_pre_innervation or not valid_ptp:
        return False

    return True


# ==================== Single-Trial Worker ====================

def _process_single_trial(trial_idx, epoch_pre_data, epoch_post_data, epoch_emg_data,
                          eeg_info, emg_info,
                          pre_events_row, post_events_row, emg_events_row,
                          pre_tmin, post_tmin, emg_tmin,
                          calibration_params, trial_reject_opts, ica, baseline,
                          artifact_window_1, artifact_window_2, leadfield, reject_range,
                          filter_opts, filter_opts_emg, emg_reject_opts, use_ica_on_pre, line_freq):
    """Process a single trial (pre, post, EMG). Returns (trial_idx, result_pre, result_post) or (trial_idx, None, None)."""
    epoch_pre = mne.EpochsArray(
        epoch_pre_data, info=eeg_info, events=pre_events_row, tmin=pre_tmin, verbose=False)
    epoch_post = mne.EpochsArray(
        epoch_post_data, info=eeg_info, events=post_events_row, tmin=post_tmin, verbose=False)
    epoch_emg = mne.EpochsArray(
        epoch_emg_data, info=emg_info, events=emg_events_row, tmin=emg_tmin, verbose=False)

    result_pre = preprocess_pre_trial(epoch_pre, calibration_params, trial_reject_opts, filter_opts, use_ica_on_pre)
    result_post = preprocess_post_trial(epoch_post, calibration_params, trial_reject_opts, ica, baseline, artifact_window_1, artifact_window_2, leadfield, reject_range)
    result_emg = preprocess_emg_trial(epoch_emg, filter_opts_emg, emg_reject_opts, calibration_params, line_freq)

    if result_pre is not False and result_post is not False and result_emg is not False:
        return trial_idx, result_pre, result_post
    return trial_idx, None, None


# ==================== Calibration persistence (two-step simulation) ====================

def _build_calibration_bundle(epochs_pre_cal, epochs_post_cal, ica, calibration_params, rejected_calibration, n_trials_use):
    return {
        'calibration_params': calibration_params,
        'ica': ica,
        'rejected_calibration': rejected_calibration,
        'epochs_pre_cal': epochs_pre_cal,
        'epochs_post_cal': epochs_post_cal,
        'n_trials_use': n_trials_use,
    }


def _save_calibration_bundle(path, bundle):
    np.save(path, bundle, allow_pickle=True)


def _load_calibration_bundle(path):
    return np.load(path, allow_pickle=True).item()


def _assert_calibration_value_equal(orig_val, load_val, path):
    if orig_val is None or load_val is None:
        assert orig_val is load_val, path
    elif isinstance(orig_val, np.ndarray):
        assert np.allclose(orig_val, load_val), path
    elif isinstance(orig_val, dict):
        assert orig_val.keys() == load_val.keys(), path
        for sub_key in orig_val:
            _assert_calibration_value_equal(orig_val[sub_key], load_val[sub_key], f'{path}.{sub_key}')
    elif isinstance(orig_val, list):
        assert len(orig_val) == len(load_val), path
        for i, (orig_item, load_item) in enumerate(zip(orig_val, load_val)):
            _assert_calibration_value_equal(orig_item, load_item, f'{path}[{i}]')
    else:
        assert orig_val == load_val, path


def _verify_calibration_bundle(original, loaded):
    """Assert round-trip integrity of the calibration bundle."""
    assert set(original.keys()) == set(loaded.keys())
    assert original['n_trials_use'] == loaded['n_trials_use']
    _assert_calibration_value_equal(original['rejected_calibration'], loaded['rejected_calibration'], 'rejected_calibration')
    assert original['ica'].exclude == loaded['ica'].exclude
    assert np.allclose(original['ica'].mixing_matrix_, loaded['ica'].mixing_matrix_)
    assert np.allclose(original['ica'].unmixing_matrix_, loaded['ica'].unmixing_matrix_)
    for orig_epochs, load_epochs in (
        (original['epochs_pre_cal'], loaded['epochs_pre_cal']),
        (original['epochs_post_cal'], loaded['epochs_post_cal']),
    ):
        assert np.allclose(orig_epochs.get_data(), load_epochs.get_data())
        assert orig_epochs.ch_names == load_epochs.ch_names
        assert np.array_equal(orig_epochs.events, load_epochs.events)
    _assert_calibration_value_equal(
        original['calibration_params'], loaded['calibration_params'], 'calibration_params')


def _load_subject_epochs(site_id, subject_id, cfg):
    data_path = _PREP_ROOT / "data_epoched" / "raw_eeglab_and_block_idents"
    subject_path = data_path / site_id / subject_id
    if not subject_path.exists():
        raise FileNotFoundError(f"Subject directory not found at {subject_path}")

    forward_path = _PREP_ROOT / "subjects_dir_fsaverage" / "fsaverage" / "fsaverage-fwd.fif"
    if not forward_path.exists():
        raise FileNotFoundError(f"Forward solution not found at {forward_path}. Run: python {_PREP_ROOT / 'build_fsaverage_forward.py'}")
    channel_order = mne.read_forward_solution(forward_path).ch_names
    montage = mne.channels.make_standard_montage('standard_1005')

    epochs = mne.read_epochs_eeglab(os.path.join(subject_path, f'{subject_id}_task-tep_all_eeg.set'))
    emg_names = [ch for ch in epochs.ch_names if 'emg' in ch.lower() or 'apb' in ch.lower() or 'fdi' in ch.lower()]
    epochs_emg = epochs.copy().pick(emg_names)
    epochs.pick(cfg.common_channels)
    epochs.reorder_channels(channel_order)
    epochs.set_montage(None)
    epochs.set_montage(montage)

    block_ids = scipy.io.loadmat(
        os.path.join(subject_path, f'{subject_id}_block_identifiers.mat'),
        simplify_cells=True,
    )['block_identifiers_trials']

    if channel_order != epochs.ch_names:
        raise ValueError(f"Channel order mismatch: {channel_order} vs {epochs.ch_names}")

    return epochs, epochs_emg, block_ids


def _compute_leadfield():
    forward_path = _PREP_ROOT / "subjects_dir_fsaverage" / "fsaverage" / "fsaverage-fwd.fif"
    forward = mne.read_forward_solution(forward_path)
    return forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0)


def _run_calibration_stage(epochs, epochs_emg, cfg, calibration_bundle_path):
    """Calibration only: writes bundle to disk; no state returned except the path."""
    dicts = cfg.to_dicts()
    channel_reject_opts = dicts['channel_reject_opts']
    ica_opts = dicts['ica_opts']
    sound_opts = dicts['sound_opts']
    ssp_sir_opts = dicts['ssp_sir_opts']
    trial_reject_opts = dicts['trial_reject_opts']
    emg_reject_opts = dicts['emg_reject_opts']
    filter_opts = dicts['filter_opts']
    filter_opts_emg = dicts['filter_opts_emg']
    leadfield = _compute_leadfield()

    n_trials_use = cfg.n_trials_goal + 25
    check_emg = epochs_emg.copy()[0:n_trials_use].crop(
        cfg.emg_time_range[0], cfg.emg_time_range[1]
    ).resample(cfg.target_sfreq, method='polyphase')
    picked_channel, emg_filter_coefficients = _select_best_emg_channel(
        check_emg, emg_reject_opts['ptp_options'], filter_opts_emg)
    epochs_emg.pick(picked_channel)

    while True:
        out = preprocess_calibration(
            epochs[0:n_trials_use], epochs_emg[0:n_trials_use],
            cfg.pre_range, cfg.post_range, cfg.baseline,
            cfg.artifact_window_1, cfg.artifact_window_2, cfg.reject_range,
            cfg.emg_time_range, channel_reject_opts, ica_opts, trial_reject_opts,
            emg_reject_opts, sound_opts, ssp_sir_opts, leadfield,
            filter_opts, filter_opts_emg, cfg.line_freq, cfg.target_sfreq,
            cfg.n_trials_goal, cfg.use_ica_on_pre, emg_filter_coefficients)
        if isinstance(out, int):
            n_trials_use += out
        else:
            break

    epochs_pre_cal, epochs_post_cal, ica, calibration_params, rejected_calibration = out
    bundle = _build_calibration_bundle(
        epochs_pre_cal, epochs_post_cal, ica, calibration_params, rejected_calibration, n_trials_use)
    _save_calibration_bundle(calibration_bundle_path, bundle)
    bundle_loaded = _load_calibration_bundle(calibration_bundle_path)
    _verify_calibration_bundle(bundle, bundle_loaded)


def _run_online_processing_stage(
    epochs, epochs_emg, block_ids, cfg, subject_output, subject_id, calibration_bundle_path,
):
    """Online trial processing: only config, epochs, and calibration bundle from disk."""
    dicts = cfg.to_dicts()
    trial_reject_opts = dicts['trial_reject_opts']
    emg_reject_opts = dicts['emg_reject_opts']
    filter_opts = dicts['filter_opts']
    filter_opts_emg = dicts['filter_opts_emg']
    leadfield = _compute_leadfield()

    bundle = _load_calibration_bundle(calibration_bundle_path)
    calibration_params = bundle['calibration_params']
    ica = bundle['ica']
    rejected_calibration = bundle['rejected_calibration']
    epochs_pre_cal = bundle['epochs_pre_cal']
    epochs_post_cal = bundle['epochs_post_cal']
    n_trials_use = bundle['n_trials_use']

    n_online = len(epochs) - n_trials_use
    all_eeg_data = epochs.get_data(copy=False)
    all_emg_data = epochs_emg.get_data(copy=False)

    # Build temporary epochs for pre/post/emg from the online trials, crop & resample in batch
    online_eeg_data = all_eeg_data[n_trials_use:]
    online_emg_data = all_emg_data[n_trials_use:]
    online_events = epochs.events[n_trials_use:]
    online_emg_events = epochs_emg.events[n_trials_use:]

    # Batch-crop and resample pre-stim epochs
    epochs_pre_batch = mne.EpochsArray(
        online_eeg_data, info=epochs.info, events=online_events, tmin=epochs.tmin, verbose=False)
    epochs_pre_batch.crop(cfg.pre_range[0], cfg.pre_range[1]).resample(cfg.target_sfreq, method='polyphase')
    pre_batch_data = epochs_pre_batch.get_data(copy=False)
    pre_batch_info = epochs_pre_batch.info
    pre_batch_tmin = epochs_pre_batch.tmin
    pre_batch_events = epochs_pre_batch.events

    # Batch-crop and resample post-stim epochs
    epochs_post_batch = mne.EpochsArray(
        online_eeg_data, info=epochs.info, events=online_events, tmin=epochs.tmin, verbose=False)
    epochs_post_batch.crop(cfg.post_range[0], cfg.post_range[1]).resample(cfg.target_sfreq, method='polyphase')
    post_batch_data = epochs_post_batch.get_data(copy=False)
    post_batch_tmin = epochs_post_batch.tmin
    post_batch_events = epochs_post_batch.events

    # Batch-crop and resample EMG epochs
    epochs_emg_batch = mne.EpochsArray(
        online_emg_data, info=epochs_emg.info, events=online_emg_events, tmin=epochs_emg.tmin, verbose=False)
    epochs_emg_batch.crop(cfg.emg_time_range[0], cfg.emg_time_range[1]).resample(cfg.target_sfreq, method='polyphase')
    emg_batch_data = epochs_emg_batch.get_data(copy=False)
    emg_batch_info = epochs_emg_batch.info
    emg_batch_tmin = epochs_emg_batch.tmin
    emg_batch_events = epochs_emg_batch.events

    # Free memory from originals
    del epochs_pre_batch, epochs_post_batch, epochs_emg_batch
    del online_eeg_data, online_emg_data

    # --- Parallel single-trial processing ---
    pre_list = [epochs_pre_cal]
    post_list = [epochs_post_cal]
    bad_trials_online = []

    def process_trial(i):
        trial_idx = n_trials_use + i
        # Use pre-prepared data (no copy needed for read, functions do get_data(copy=True) internally)
        epoch_pre_data = pre_batch_data[i:i+1]
        epoch_post_data = post_batch_data[i:i+1]
        epoch_emg_data = emg_batch_data[i:i+1]

        return _process_single_trial(
            trial_idx, epoch_pre_data, epoch_post_data, epoch_emg_data,
            pre_batch_info, emg_batch_info,
            pre_batch_events[i:i+1], post_batch_events[i:i+1], emg_batch_events[i:i+1],
            pre_batch_tmin, post_batch_tmin, emg_batch_tmin,
            calibration_params, trial_reject_opts, ica, cfg.baseline,
            cfg.artifact_window_1, cfg.artifact_window_2, leadfield, cfg.reject_range,
            filter_opts, filter_opts_emg, emg_reject_opts, cfg.use_ica_on_pre, cfg.line_freq)

    # Use ThreadPoolExecutor - ICA and MNE operations hold GIL but numpy releases it
    # Process sequentially to maintain deterministic ordering (same as original)
    # but with pre-prepared epochs for speed
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_trial, i): i for i in range(n_online)}
        for future in futures:
            results.append((futures[future], future.result()))

    # Sort by original order to maintain determinism
    results.sort(key=lambda x: x[0])

    for i, (_, result) in enumerate(results):
        trial_idx = n_trials_use + i
        _, result_pre, result_post = result
        if result_pre is not None:
            pre_list.append(result_pre)
            post_list.append(result_post)
        else:
            bad_trials_online.append(trial_idx)

    epochs_pre_final = mne.concatenate_epochs(pre_list)
    epochs_post_final = mne.concatenate_epochs(post_list)

    # Block identifiers bookkeeping
    bad_keys = ['bad_trials_ocular', 'bad_trials_pre', 'bad_trials_post', 'bad_trials_emg']
    block_ids_cal = block_ids[:n_trials_use]
    for key in bad_keys:
        good = np.array([i for i in range(len(block_ids_cal)) if i not in rejected_calibration[key]])
        block_ids_cal = block_ids_cal[good]
    good_online = np.array([i for i in range(n_trials_use, len(epochs)) if i not in bad_trials_online])
    block_ids_online = block_ids[good_online]
    block_ids_final = np.concatenate((block_ids_cal, block_ids_online))

    # --- Save results ---
    for epoch, label in zip([epochs_pre_final, epochs_post_final], ['pre', 'post']):
        epoch.save(os.path.join(subject_output, f"{subject_id}_{label}.fif"), overwrite=True)

    np.save(os.path.join(subject_output, f'{subject_id}_block_identifiers.npy'), block_ids_final)


def run_subject_processing(site_id: str, subject_id: str):
    """Main preprocessing pipeline for a single subject."""
    cfg = get_default_config()
    output_path = _PREP_ROOT / f"data_processed_pre_ica_{cfg.use_ica_on_pre}_v4"
    subject_output = output_path / subject_id
    os.makedirs(subject_output, exist_ok=True)

    calibration_bundle_path = subject_output / f'{subject_id}_calibration_bundle.npy'

    epochs, epochs_emg, block_ids = _load_subject_epochs(site_id, subject_id, cfg)
    _run_calibration_stage(epochs, epochs_emg, cfg, calibration_bundle_path)

    cfg = get_default_config()
    _run_online_processing_stage(
        epochs, epochs_emg, block_ids, cfg, subject_output, subject_id, calibration_bundle_path)


# %%
def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for a single subject.")
    parser.add_argument("--site", required=True, type=str, help="Site identifier.")
    parser.add_argument("--subject", required=True, type=str, help="Subject identifier.")
    args = parser.parse_args()
    run_subject_processing(site_id=args.site, subject_id=args.subject)


if __name__ == "__main__":
    main()
