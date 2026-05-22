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

from ica_calibrator import get_number_of_components, get_ica
from ssp_sir_python import ssp_sir_to_average, ssp_sir_trials, ssp_sir_single_trial
from sound_modified import sound
from channel_interpolations import custom_get_interpolation_matrix, apply_channel_interpolation

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
    info = {}
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
    info['pre_stim_filter'] = pre_filter_coefficients
    bad_channels_post = _detect_bad_channels_post(epochs_post, channel_reject_opts['post'], reject_range)
    bad_channels = list(np.union1d(bad_channels_pre, bad_channels_post))

    info['channels_before_rejection'] = epochs_post.ch_names
    info['bad_channels_pre'] = bad_channels_pre
    info['bad_channels_post'] = bad_channels_post
    info['bad_channels'] = bad_channels

    if bad_channels:
        epochs_pre, interpolation_info = _interpolate_bad_channels(epochs_pre, bad_channels, None)
        epochs_pre_ica, _ = _interpolate_bad_channels(epochs_pre_ica, bad_channels, interpolation_info)
        epochs_post, _ = _interpolate_bad_channels(epochs_post, bad_channels, interpolation_info)
    else:
        interpolation_info = None
    info['channel_interpolation_info'] = interpolation_info

    # ICA calibration on filtered pre-stim
    ica_data = epochs_pre_ica.get_data(copy=True)
    ica_data, ica_filter_coefficients = _butter_filter(ica_data, ica_opts['filtering']['cutoff'], 'bandpass',
                                               epochs_pre_ica.info['sfreq'], ica_opts['filtering']['order_bandpass'],
                                               ica_opts['filtering']['pad_time_bandpass'])
    info['epochs_pre_ica_filter'] = ica_filter_coefficients
    epochs_pre_ica_filtered = mne.EpochsArray(ica_data, info=epochs_pre_ica.info, events=epochs_pre_ica.events, tmin=epochs_pre_ica.times[0])

    epochs_post.set_eeg_reference('average', projection=False, verbose=False)
    epochs_pre_ica_filtered.set_eeg_reference('average', projection=False, verbose=False)

    n_components = get_number_of_components(epochs_pre_ica_filtered.get_data(copy=True), ica_opts['pc_threshold'])
    ica, excluded_components, ica_component_labels = get_ica(epochs_pre_ica_filtered, n_components, None,
                                             ica_opts['bad_component_thresholds'],
                                             ica_opts['n_min_comps_to_reject'],
                                             ica_opts['threshold_min_components_to_reject'])
    info['ica_comps_excluded'] = excluded_components
    info['ic_label_dict'] = ica_component_labels
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
        bad_ocular_pre, ocular_threshold_pre = _detect_ocular_trials(
            ica, epochs_pre, trial_reject_opts['ocular']['pre_timerange_min'], None,
            excluded_components['eye blink'], trial_reject_opts['ocular']['z_thresh'])
        info['bad_trials_ocular_pre'] = bad_ocular_pre
        info['ocular_thresholds_pre'] = ocular_threshold_pre

    bad_ocular_post, ocular_threshold_post = _detect_ocular_trials(
        ica, epochs_post, trial_reject_opts['ocular']['post_timerange'][0],
        trial_reject_opts['ocular']['post_timerange'][1],
        excluded_components['eye blink'], trial_reject_opts['ocular']['z_thresh'])

    if use_ica_on_pre:
        bad_ocular = list(np.union1d(bad_ocular_pre, bad_ocular_post).astype(int))
    else:
        bad_ocular = bad_ocular_post

    info['bad_trials_ocular_post'] = bad_ocular_post
    info['ocular_thresholds_post'] = ocular_threshold_post
    info['bad_trials_ocular'] = bad_ocular
    info['trials_before_ocular_rejection'] = epochs_pre.get_data(copy=True).shape[0]

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
    info['trials_before_pre_eeg_rejection'] = epochs_pre.get_data(copy=True).shape[0]
    info['bad_trials_pre'] = bad_pre
    info['good_trial_stats_pre'] = stats_pre
    epochs_pre, epochs_post, epochs_emg = _drop_bad_trials([epochs_pre, epochs_post, epochs_emg], bad_pre)

    # Apply SOUND to post-stim
    epochs_post.apply_baseline(baseline).set_eeg_reference('average', projection=False, verbose=False)
    post_t0 = epochs_post.times[epochs_post.times > 0][0]
    info['post_mintime'] = post_t0
    epochs_post.crop(post_t0, None)

    post_data = epochs_post.get_data(copy=True)
    evoked = np.mean(post_data, axis=0)
    n_channels = evoked.shape[0]
    sound_filter, sound_sigmas, n_iters, converged = sound(
        evoked.T, 0, np.ones((n_channels, 1)), n_channels, leadfield,
        sound_opts['max_iterations'], sound_opts['lambda'],
        sound_opts['convergence_tolerance'], sound_opts['fixed_max_iterations'])

    for i in range(post_data.shape[0]):
        post_data[i, :, :] = np.matmul(sound_filter, post_data[i, :, :])

    epochs_post = mne.EpochsArray(post_data, epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0])
    epochs_post.set_eeg_reference('average', projection=False, verbose=False)
    info['sound_filter'] = sound_filter
    info['n_iters_sound'] = n_iters
    info['sound_convergence_reached'] = converged
    info['sound_sigmas'] = sound_sigmas

    # Apply SSP-SIR to post-stim
    post_data = epochs_post.get_data(copy=True)
    evoked = np.mean(post_data, axis=0)
    corrected, artifact_topographies, suppressed, kernel, P, M, PL, projection_suppression, projection_original, n_pcs = ssp_sir_to_average(
        evoked, leadfield, epochs_post.info['sfreq'], ssp_sir_opts['timerange'], method=ssp_sir_opts['method'])

    post_data = ssp_sir_trials(post_data, P, projection_suppression, projection_original, kernel)
    epochs_post = mne.EpochsArray(post_data, epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0])

    info['sspsir_suppression_matrix_P'] = P
    info['sspsir_suppression_matrix_PL'] = PL
    info['sspsir_filter_kernel'] = kernel
    info['sspsir_M'] = M
    info['sspsir_n_pcs_removed'] = n_pcs
    info['sspsir_data_corrected_ave'] = corrected
    info['sspsir_artifact_topographies'] = artifact_topographies
    info['sspsir_data_suppressed_ave'] = suppressed
    info['sspsir_sir_projmat_suppr'] = projection_suppression
    info['sspsir_sir_projmat_orig'] = projection_original

    # Second artifact interpolation
    epochs_post = mne.preprocessing.fix_stim_artifact(epochs_post, tmin=post_t0, tmax=artifact_window_2[1], mode='window')
    epochs_post.set_eeg_reference('average', projection=False, verbose=False)

    # Detect bad post-stim trials
    bad_post, stats_post = _find_bad_trials(
        epochs_post.copy().crop(reject_range[0], reject_range[1]),
        trial_reject_opts['post']['global_zscore_threshold'],
        trial_reject_opts['post']['local_zscore_threshold'], False, False)
    info['trials_before_post_eeg_rejection'] = epochs_post.get_data(copy=True).shape[0]
    info['bad_trials_post'] = bad_post
    info['good_trial_stats_post'] = stats_post
    epochs_pre, epochs_post, epochs_emg = _drop_bad_trials([epochs_pre, epochs_post, epochs_emg], bad_post)

    # EMG processing
    emg_data = epochs_emg.get_data(copy=True)
    emg_data = _apply_filter(emg_data, emg_filter_coefficients, filter_opts_emg['pad_time'], epochs_emg.info['sfreq'])
    epochs_emg = mne.EpochsArray(emg_data, info=epochs_emg.info, events=epochs_emg.events, tmin=epochs_emg.times[0])
    bad_emg, epochs_emg, emg_time_info = _detect_bad_emg_trials(
        epochs_emg, emg_reject_opts['pre_innervation_options'], emg_reject_opts['ptp_options'], line_freq)

    info['emg_filter'] = emg_filter_coefficients
    info['emg_prep_times'] = emg_time_info
    info['bad_trials_emg'] = bad_emg
    info['trials_before_emg_rejection'] = epochs_emg.get_data(copy=True).shape[0]

    epochs_pre, epochs_post, epochs_emg = _drop_bad_trials([epochs_pre, epochs_post, epochs_emg], bad_emg)
    info['n_trials_left'] = epochs_pre.get_data(copy=True).shape[0]

    if info['n_trials_left'] < n_trials_goal:
        return n_trials_goal - info['n_trials_left']

    return epochs_pre, epochs_post, ica, info


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


# ==================== Subject-Level Runner ====================

def run_subject_processing(site_id: str, subject_id: str):
    """Main preprocessing pipeline for a single subject."""
    # Paths
    data_path = _PREP_ROOT / "data_epoched" / "raw_eeglab_and_block_idents"
    use_ica_on_pre = False
    output_path = _PREP_ROOT / f"data_processed_pre_ica_{use_ica_on_pre}_v4"

    # Time ranges
    pre_range = [-0.505, -0.005]
    post_range = [-0.03, 0.1]
    baseline = [-0.025, -0.015]
    reject_range = [0.02, 0.06]
    artifact_window_1 = [-0.014, 0.014]
    artifact_window_2 = [None, 0.015]
    emg_time_range = [-0.5, 0.2]

    # Channel rejection
    freq_range = [30, 47]
    channel_reject_opts = {
        'pre': {'z_score_threshold_mad': [-3, 3], 'z_score_threshold_power': 5, 'fmin_fmax': freq_range, 'z_score_threshold_autocorr': 4},
        'post': {'z_score_threshold_auc': 3},
    }

    # ICA
    ica_opts = {
        'pc_threshold': 0.99,
        'bad_component_thresholds': {'eye blink': 0.9},
        'n_min_comps_to_reject': {'eye blink': 2},
        'threshold_min_components_to_reject': {'eye blink': 0.7},
        'pre_timerange': [-1.1, -0.005],
        'filtering': {'order_bandpass': 2, 'order_bandstop': 2, 'pad_time_bandpass': 0.5, 'cutoff': [1, 100]},
    }

    # SOUND & SSP-SIR
    sound_opts = {'max_iterations': 10, 'lambda': 0.01, 'convergence_tolerance': 1e-9, 'fixed_max_iterations': False}
    ssp_sir_opts = {'timerange': ['automatic', 50], 'method': ['threshold', 0.99]}

    # Trial rejection
    trial_reject_opts = {
        'ocular': {'z_thresh': 2, 'pre_timerange_min': -0.1, 'post_timerange': [0.015, None]},
        'pre': {'global_zscore_threshold': [-8, 4], 'local_zscore_threshold': 5, 'psd_zscore_threshold': False, 'psd_freq_range': freq_range},
        'post': {'global_zscore_threshold': [-np.inf, 6], 'local_zscore_threshold': 5},
    }

    # EMG rejection
    pre_innervation_opts = {'tmin': -0.2, 'tmax': -0.015, 'threshold': 50e-6}
    ptp_opts = {'tmin': 0.02, 'tmax': 0.05, 'min_ptp_height': 50e-6, 'min_distance': 0.005, 'prominence': 10e-6, 'check_ptp': False}
    emg_reject_opts = {'pre_innervation_options': pre_innervation_opts, 'ptp_options': ptp_opts}

    # Channel setup and forward model
    common_channels = ['AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8']
    montage = mne.channels.make_standard_montage('standard_1005')
    forward_path = _PREP_ROOT / "subjects_dir_fsaverage" / "fsaverage" / "fsaverage-fwd.fif"
    if not forward_path.exists():
        raise FileNotFoundError(f"Forward solution not found at {forward_path}. Run: python {_PREP_ROOT / 'build_fsaverage_forward.py'}")
    forward = mne.read_forward_solution(forward_path)
    leadfield = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0)
    channel_order = forward.ch_names

    # Filter options
    filter_opts = {'cutoff': [2, 47], 'btype': 'bandpass', 'order': 2, 'pad_time': 0.1}
    filter_opts_emg = {'cutoff': 2, 'btype': 'highpass', 'order': 4, 'pad_time': 0.5}

    # General parameters
    line_freq = 50
    target_sfreq = 1000
    n_trials_goal = 100

    # --- Load data ---
    subject_path = data_path / site_id / subject_id
    subject_output = output_path / subject_id

    if not subject_path.exists():
        raise FileNotFoundError(f"Subject directory not found at {subject_path}")

    os.makedirs(subject_output, exist_ok=True)

    epochs = mne.read_epochs_eeglab(os.path.join(subject_path, f'{subject_id}_task-tep_all_eeg.set'))
    emg_names = [ch for ch in epochs.ch_names if 'emg' in ch.lower() or 'apb' in ch.lower() or 'fdi' in ch.lower()]
    epochs_emg = epochs.copy().pick(emg_names)
    epochs.pick(common_channels)
    epochs.reorder_channels(channel_order)
    epochs.set_montage(None)
    epochs.set_montage(montage)

    block_ids = scipy.io.loadmat(os.path.join(subject_path, f'{subject_id}_block_identifiers.mat'), simplify_cells=True)['block_identifiers_trials']

    if channel_order != epochs.ch_names:
        raise ValueError(f"Channel order mismatch: {channel_order} vs {epochs.ch_names}")

    # --- Calibration ---
    n_trials_use = n_trials_goal + 25
    check_emg = epochs_emg.copy()[0:n_trials_use].crop(emg_time_range[0], emg_time_range[1]).resample(target_sfreq, method='polyphase')
    picked_channel, emg_filter_coefficients = _select_best_emg_channel(check_emg, ptp_opts, filter_opts_emg)
    epochs_emg.pick(picked_channel)

    while True:
        out = preprocess_calibration(
            epochs[0:n_trials_use], epochs_emg[0:n_trials_use],
            pre_range, post_range, baseline,
            artifact_window_1, artifact_window_2, reject_range,
            emg_time_range, channel_reject_opts, ica_opts, trial_reject_opts,
            emg_reject_opts, sound_opts, ssp_sir_opts, leadfield,
            filter_opts, filter_opts_emg, line_freq, target_sfreq,
            n_trials_goal, use_ica_on_pre, emg_filter_coefficients)
        if isinstance(out, int):
            n_trials_use += out
        else:
            break

    epochs_pre_cal, epochs_post_cal, ica, preprocessing_info = out

    preprocessing_info['n_trials_calibration'] = len(epochs_pre_cal)
    preprocessing_info['used_emg_channel'] = picked_channel
    preprocessing_info['n_trials_used_in_calibration'] = n_trials_use
    preprocessing_info['n_trials_original'] = len(epochs)

    # --- Single-trial processing ---
    pre_list = [epochs_pre_cal]
    post_list = [epochs_post_cal]
    bad_trials_online = []
    n_total = len(epochs) - n_trials_use

    for i, trial_idx in enumerate(range(n_trials_use, len(epochs))):
        print(f"Trial {i + 1}/{n_total}")
        epoch = mne.EpochsArray(
            epochs.get_data(item=trial_idx), info=epochs.info,
            events=epochs.events[trial_idx:trial_idx + 1], tmin=epochs.tmin, verbose=False)
        epoch_emg = mne.EpochsArray(
            epochs_emg.get_data(item=trial_idx), info=epochs_emg.info,
            events=epochs_emg.events[trial_idx:trial_idx + 1], tmin=epochs_emg.tmin, verbose=False)

        # Pre-stim
        epoch_pre = epoch.copy().crop(pre_range[0], pre_range[1]).resample(target_sfreq, method='polyphase')
        result_pre = preprocess_pre_trial(epoch_pre, preprocessing_info, trial_reject_opts, filter_opts, use_ica_on_pre)

        # Post-stim
        epoch_post = epoch.copy().crop(post_range[0], post_range[1]).resample(target_sfreq, method='polyphase')
        result_post = preprocess_post_trial(epoch_post, preprocessing_info, trial_reject_opts, ica, baseline, artifact_window_1, artifact_window_2, leadfield, reject_range)

        # EMG
        epoch_emg.crop(emg_time_range[0], emg_time_range[1]).resample(target_sfreq, method='polyphase')
        result_emg = preprocess_emg_trial(epoch_emg, filter_opts_emg, emg_reject_opts, preprocessing_info, line_freq)

        if result_pre is not False and result_post is not False and result_emg is not False:
            pre_list.append(result_pre)
            post_list.append(result_post)
        else:
            bad_trials_online.append(trial_idx)

    preprocessing_info['bad_trials_calibrated'] = bad_trials_online
    epochs_pre_final = mne.concatenate_epochs(pre_list)
    epochs_post_final = mne.concatenate_epochs(post_list)

    # Block identifiers bookkeeping
    bad_keys = ['bad_trials_ocular', 'bad_trials_pre', 'bad_trials_post', 'bad_trials_emg']
    block_ids_cal = block_ids[:n_trials_use]
    for key in bad_keys:
        good = np.array([i for i in range(len(block_ids_cal)) if i not in preprocessing_info[key]])
        block_ids_cal = block_ids_cal[good]
    good_online = np.array([i for i in range(n_trials_use, len(epochs)) if i not in bad_trials_online])
    block_ids_online = block_ids[good_online]
    block_ids_final = np.concatenate((block_ids_cal, block_ids_online))
    preprocessing_info['block_identifiers'] = block_ids_final
    preprocessing_info['n_trials_final'] = len(epochs_pre_final)

    # --- Save results ---
    for epoch, label in zip([epochs_pre_final, epochs_post_final], ['pre', 'post']):
        epoch.save(os.path.join(subject_output, f"{subject_id}_{label}.fif"), overwrite=True)

    np.savez(os.path.join(subject_output, f'{subject_id}_preprocessing_info.npz'), **preprocessing_info)
    np.save(os.path.join(subject_output, f'{subject_id}_block_identifiers.npy'), block_ids_final)


# %%
def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for a single subject.")
    parser.add_argument("--site", required=True, type=str, help="Site identifier.")
    parser.add_argument("--subject", required=True, type=str, help="Subject identifier.")
    args = parser.parse_args()
    run_subject_processing(site_id=args.site, subject_id=args.subject)


if __name__ == "__main__":
    main()
