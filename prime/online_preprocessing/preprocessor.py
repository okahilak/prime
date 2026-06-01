"""
Preprocessor: stateful object that accumulates trials, runs calibration, and
preprocesses individual trials using the resulting calibration parameters.

Usage
-----
    preprocessor = Preprocessor(forward_path, source_sfreq, info)
    preprocessor.add_raw_pre_epoch(raw_pre)   # (n_samples, n_channels)
    preprocessor.add_raw_post_epoch(raw_post)

    cal_trials = preprocessor.calibrate()  # list[ProcessedTrial]

    processed = preprocessor.preprocess(raw_pre, raw_post)  # None if rejected
    processed.epoch_pre, processed.epoch_post
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from scipy.stats import median_abs_deviation, zscore
from scipy.signal import butter, filtfilt


@dataclass(frozen=True)
class ProcessedTrial:
    """Result of preprocessing a single trial (pre and post epochs)."""
    epoch_pre: mne.EpochsArray
    epoch_post: mne.EpochsArray

from prime_config import (
    epoch_n_times,
    get_post_epoch_time_range,
    get_pre_epoch_time_range,
    get_raw_post_epoch_time_range,
    get_raw_pre_epoch_time_range,
)
from online_preprocessing.utils.ica_calibrator import get_number_of_components, get_ica
from online_preprocessing.utils.ssp_sir_python import (
    ssp_sir_to_average,
    ssp_sir_trials,
    ssp_sir_single_trial,
)
from online_preprocessing.utils.sound_modified import sound
from online_preprocessing.utils.channel_interpolations import (
    custom_get_interpolation_matrix,
    apply_channel_interpolation,
)
from online_preprocessing.config import get_default_config

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


def _validate_time_range_within(
    name: str,
    tmin: float,
    tmax: float,
    outer_tmin: float,
    outer_tmax: float,
) -> None:
    if not (outer_tmin <= tmin and tmax <= outer_tmax and tmin < tmax):
        raise ValueError(
            f"{name} window [{tmin}, {tmax}] must satisfy "
            f"{outer_tmin} <= tmin < tmax <= {outer_tmax}"
        )


def _validate_pre_epoch_window(
    pre_epoch_tmin: float,
    pre_epoch_tmax: float,
    raw_pre_epoch_tmin: float,
    raw_pre_epoch_tmax: float,
) -> None:
    _validate_time_range_within(
        "Pre epoch", pre_epoch_tmin, pre_epoch_tmax, raw_pre_epoch_tmin, raw_pre_epoch_tmax,
    )


def _validate_post_epoch_window(
    post_epoch_tmin: float,
    post_epoch_tmax: float,
    raw_post_epoch_tmin: float,
    raw_post_epoch_tmax: float,
) -> None:
    if not (
        raw_post_epoch_tmin <= post_epoch_tmin
        and post_epoch_tmax <= raw_post_epoch_tmax
        and post_epoch_tmin < post_epoch_tmax
    ):
        raise ValueError(
            f"Post epoch window [{post_epoch_tmin}, {post_epoch_tmax}] must satisfy "
            f"{raw_post_epoch_tmin} <= post_epoch_tmin < post_epoch_tmax <= {raw_post_epoch_tmax}"
        )


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


def _numpy_to_epochs_array(
    data: np.ndarray,
    info: mne.Info,
    tmin: float,
) -> mne.EpochsArray:
    """Convert (n_samples, n_channels) buffer to a single-trial EpochsArray."""
    data = np.asarray(data, dtype=np.float64)
    return mne.EpochsArray(
        data.T[np.newaxis, :, :],
        info=info,
        events=np.array([[0, 0, 1]]),
        tmin=tmin,
        verbose=False,
    )


def _epochs_to_buffer(epochs: mne.BaseEpochs) -> np.ndarray:
    """MNE epoch data as (n_samples, n_channels)."""
    return epochs.get_data(copy=False)[0].T


def crop_eeg_buffer(
    eeg_buffer: np.ndarray,
    time_offsets: np.ndarray,
    tmin: float,
    tmax: float,
) -> np.ndarray:
    """Crop (n_samples, n_channels) to [tmin, tmax] inclusive relative to the event."""
    mask = (time_offsets >= tmin) & (time_offsets <= tmax)
    return eeg_buffer[mask]


def crop_mne_trial_to_raw_epochs(
    trial: mne.BaseEpochs,
    raw_pre_tmin: float,
    raw_pre_tmax: float,
    raw_post_tmin: float,
    raw_post_tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a full-trial EpochsArray into raw pre/post numpy buffers."""
    raw_pre = _epochs_to_buffer(
        trial.copy().crop(raw_pre_tmin, raw_pre_tmax, include_tmax=True),
    )
    raw_post = _epochs_to_buffer(
        trial.copy().crop(raw_post_tmin, raw_post_tmax, include_tmax=True),
    )
    return raw_pre, raw_post


def append_calibration_epochs(
    epochs_pre,
    epochs_pre_ica,
    epochs_post,
    raw_pre: np.ndarray,
    raw_post: np.ndarray,
    info: mne.Info,
    source_sfreq: float,
    cfg,
    raw_pre_epoch_tmin: float,
    raw_pre_epoch_tmax: float,
    raw_post_epoch_tmin: float,
    raw_post_epoch_tmax: float,
):
    """Append one raw pre/post pair (resampled) to calibration epoch structs."""
    trial_full_pre = _numpy_to_epochs_array(raw_pre, info, raw_pre_epoch_tmin)
    trial_post = _numpy_to_epochs_array(raw_post, info, raw_post_epoch_tmin)
    trial_pre = trial_full_pre.copy().crop(cfg.pre_stim_timerange[0], cfg.pre_stim_timerange[1])
    trial_pre_ica = trial_full_pre.copy().crop(
        cfg.ica_opts.pre_timerange[0], cfg.ica_opts.pre_timerange[1],
    )
    for segment in (trial_pre, trial_pre_ica, trial_post):
        segment.resample(cfg.target_sfreq, method='polyphase')
    if epochs_pre is None:
        return trial_pre, trial_pre_ica, trial_post
    epochs_pre = mne.concatenate_epochs([epochs_pre, trial_pre])
    epochs_pre_ica = mne.concatenate_epochs([epochs_pre_ica, trial_pre_ica])
    epochs_post = mne.concatenate_epochs([epochs_post, trial_post])
    return epochs_pre, epochs_pre_ica, epochs_post


# ==================== Calibration pipeline ====================


def preprocess_calibration(epochs_pre, epochs_pre_ica, epochs_post, cfg, opts, forward):
    """Full calibration preprocessing pipeline for pre-stim and post-stim epochs."""
    channel_reject_opts = opts['channel_reject_opts']
    ica_opts = opts['ica_opts']
    trial_reject_opts = opts['trial_reject_opts']
    sound_opts = opts['sound_opts']
    ssp_sir_opts = opts['ssp_sir_opts']
    filter_opts = opts['filter_opts']

    leadfield = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0)

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
    return calibration_params, n_successful_trials, epochs_pre, epochs_post


# ==================== Single-trial processing ====================

def preprocess_pre_trial(epoch_pre, calibration_params, cfg):
    """Preprocess a single pre-stimulus trial using calibrated parameters."""
    dicts = cfg.to_dicts()
    trial_reject_opts = dicts['trial_reject_opts']
    filter_opts = dicts['filter_opts']

    if calibration_params['bad_channels']:
        epoch_pre, _ = _interpolate_bad_channels(
            epoch_pre, calibration_params['bad_channels'],
            calibration_params['channel_interpolation_info'])

    if cfg.use_ica_on_pre:
        raise NotImplementedError(
            "Applying ICA to pre-stimulus in single-trials is not currently implemented")

    data = epoch_pre.get_data(copy=True)

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
    if np.any(np.abs(z_local) > trial_reject_opts['pre']['local_zscore_threshold']):
        return False

    return mne.EpochsArray(data, info=epoch_pre.info, events=epoch_pre.events,
                           tmin=epoch_pre.times[0], verbose=False)


def preprocess_post_trial(epoch_post, calibration_params, cfg):
    """Preprocess a single post-stimulus trial using calibrated parameters."""
    dicts = cfg.to_dicts()
    trial_reject_opts = dicts['trial_reject_opts']

    epoch_post.apply_baseline(cfg.baseline)
    epoch_post = mne.preprocessing.fix_stim_artifact(
        epoch_post, tmin=cfg.artifact_window_1[0], tmax=cfg.artifact_window_1[1], mode='window')

    if calibration_params['bad_channels']:
        epoch_post, _ = _interpolate_bad_channels(
            epoch_post, calibration_params['bad_channels'],
            calibration_params['channel_interpolation_info'])

    epoch_post.set_eeg_reference('average', projection=False, verbose=False)

    ica = calibration_params['ica']

    # Check ocular ICA components
    source_time_course = ica.get_sources(epoch_post)
    source_data = source_time_course.get_data(copy=True)
    for component_idx in calibration_params['ocular_thresholds_post']:
        component_info = calibration_params['ocular_thresholds_post'][component_idx]
        z_comp = (
            np.abs(source_data[0, component_idx, component_info['time_indices_of_interest']])
            - component_info['mean']
        ) / component_info['std']
        if np.median(z_comp) > trial_reject_opts['ocular']['z_threshold']:
            return False

    ica.apply(epoch_post)

    epoch_post.apply_baseline(cfg.baseline).set_eeg_reference('average', projection=False, verbose=False)
    epoch_post.crop(calibration_params['post_mintime'], None)

    # Apply SOUND
    data = epoch_post.get_data(copy=True)
    data = np.matmul(calibration_params['sound_filter'], data)
    epoch_post = mne.EpochsArray(data, epoch_post.info, events=epoch_post.events,
                                 tmin=epoch_post.times[0], verbose=False)
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)

    # Apply SSP-SIR
    data = epoch_post.get_data(copy=True)
    data = ssp_sir_single_trial(
        data[0, :, :], calibration_params['sspsir_suppression_matrix_P'],
        calibration_params['sspsir_sir_projmat_suppr'], calibration_params['sspsir_sir_projmat_orig'],
        calibration_params['sspsir_filter_kernel'])
    epoch_post = mne.EpochsArray(data.reshape(1, data.shape[0], data.shape[1]),
                                 epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0])

    epoch_post = mne.preprocessing.fix_stim_artifact(
        epoch_post, tmin=calibration_params['post_mintime'], tmax=cfg.artifact_window_2[1],
        mode='window')
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
    if np.any(np.abs(z_local) > trial_reject_opts['post']['local_zscore_threshold']):
        return False

    return epoch_post


# ==================== Preprocessor class ====================

class Preprocessor:
    """Accumulates raw trials and produces calibration parameters on demand."""

    def __init__(self, forward_path, source_sfreq: float, info: mne.Info):
        cfg = get_default_config()
        raw_pre_epoch_tmin, raw_pre_epoch_tmax = get_raw_pre_epoch_time_range()
        raw_post_epoch_tmin, raw_post_epoch_tmax = get_raw_post_epoch_time_range()
        pre_epoch_tmin, pre_epoch_tmax = get_pre_epoch_time_range()
        post_epoch_tmin, post_epoch_tmax = get_post_epoch_time_range()
        _validate_time_range_within(
            "Pre-stim preprocessing",
            cfg.pre_stim_timerange[0],
            cfg.pre_stim_timerange[1],
            raw_pre_epoch_tmin,
            raw_pre_epoch_tmax,
        )
        _validate_time_range_within(
            "ICA",
            cfg.ica_opts.pre_timerange[0],
            cfg.ica_opts.pre_timerange[1],
            raw_pre_epoch_tmin,
            raw_pre_epoch_tmax,
        )
        _validate_pre_epoch_window(
            pre_epoch_tmin, pre_epoch_tmax, raw_pre_epoch_tmin, raw_pre_epoch_tmax,
        )
        _validate_post_epoch_window(
            post_epoch_tmin, post_epoch_tmax, raw_post_epoch_tmin, raw_post_epoch_tmax,
        )
        self._cfg = cfg
        self._info = info
        self._source_sfreq = float(source_sfreq)
        self._raw_pre_epoch_tmin = raw_pre_epoch_tmin
        self._raw_pre_epoch_tmax = raw_pre_epoch_tmax
        self._raw_post_epoch_tmin = raw_post_epoch_tmin
        self._raw_post_epoch_tmax = raw_post_epoch_tmax
        self._raw_pre_n_times = epoch_n_times(
            raw_pre_epoch_tmin, raw_pre_epoch_tmax, self._source_sfreq,
        )
        self._raw_post_n_times = epoch_n_times(
            raw_post_epoch_tmin, raw_post_epoch_tmax, self._source_sfreq,
        )
        self._pre_stim_tmin = cfg.pre_stim_timerange[0]
        self._pre_stim_tmax = cfg.pre_stim_timerange[1]
        self._pre_epoch_tmin = pre_epoch_tmin
        self._pre_epoch_tmax = pre_epoch_tmax
        self._post_epoch_tmin = post_epoch_tmin
        self._post_epoch_tmax = post_epoch_tmax
        self._opts = cfg.to_dicts()

        self._epochs_pre = None
        self._epochs_pre_ica = None
        self._epochs_post = None
        self._calibration_params = None
        self._pending_pre = None
        self._awaiting_post = False

        self._forward = mne.read_forward_solution(str(forward_path), verbose=False)
        # TODO: Should we pick the common channels here? Note that it's done in the dipole fitter.

    def _validate_raw_pre_shape(self, data: np.ndarray) -> None:
        data = np.asarray(data)
        if data.shape != (self._raw_pre_n_times, len(self._info.ch_names)):
            raise ValueError(
                f"raw pre epoch must have shape ({self._raw_pre_n_times}, {len(self._info.ch_names)}), "
                f"got {data.shape}"
            )

    def _validate_raw_post_shape(self, data: np.ndarray) -> None:
        data = np.asarray(data)
        if data.shape != (self._raw_post_n_times, len(self._info.ch_names)):
            raise ValueError(
                f"raw post epoch must have shape ({self._raw_post_n_times}, {len(self._info.ch_names)}), "
                f"got {data.shape}"
            )

    def _crop_pre_to_model_window(self, epoch_pre: mne.Epochs) -> mne.Epochs:
        return epoch_pre.copy().crop(self._pre_epoch_tmin, self._pre_epoch_tmax, include_tmax=True)

    def _crop_post_to_tep_window(self, epoch_post: mne.Epochs) -> mne.Epochs:
        return epoch_post.copy().crop(self._post_epoch_tmin, self._post_epoch_tmax, include_tmax=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_raw_pre_epoch(self, raw_pre: np.ndarray) -> None:
        """Buffer one raw pre-stimulus epoch for calibration.

        Parameters
        ----------
        raw_pre : np.ndarray, shape (n_samples, n_channels)
            EEG for ``[raw_pre_epoch_tmin, raw_pre_epoch_tmax]`` at ``source_sfreq``.
        """
        if self._awaiting_post:
            raise RuntimeError(
                "add_raw_post_epoch must be called before the next add_raw_pre_epoch"
            )
        self._validate_raw_pre_shape(raw_pre)
        self._pending_pre = np.asarray(raw_pre, dtype=np.float64)
        self._awaiting_post = True

    def add_raw_post_epoch(self, raw_post: np.ndarray) -> None:
        """Append the paired pre/post epochs to the calibration buffers.

        Parameters
        ----------
        raw_post : np.ndarray, shape (n_samples, n_channels)
            EEG for ``[raw_post_epoch_tmin, raw_post_epoch_tmax]`` at ``source_sfreq``.
        """
        if not self._awaiting_post:
            raise RuntimeError("add_raw_pre_epoch must be called before add_raw_post_epoch")
        self._validate_raw_post_shape(raw_post)
        self._epochs_pre, self._epochs_pre_ica, self._epochs_post = append_calibration_epochs(
            self._epochs_pre, self._epochs_pre_ica, self._epochs_post,
            self._pending_pre, raw_post,
            self._info, self._source_sfreq, self._cfg,
            self._raw_pre_epoch_tmin, self._raw_pre_epoch_tmax,
            self._raw_post_epoch_tmin, self._raw_post_epoch_tmax,
        )
        self._pending_pre = None
        self._awaiting_post = False

    def calibrate(self):
        """Run calibration on the accumulated trials.

        Stores the resulting parameters internally.  Call ``preprocess`` on
        individual trials afterwards.

        Returns
        -------
        list of ProcessedTrial
            Calibration trials that survived artifact rejection.
        """
        if self._awaiting_post:
            raise RuntimeError("add_raw_post_epoch was not called for the last pre epoch")
        if self._epochs_pre is None:
            raise RuntimeError("No trials have been added yet.")

        calibration_params, n_successful_trials, pre_epochs, post_epochs = preprocess_calibration(
            self._epochs_pre.copy(),
            self._epochs_pre_ica.copy(),
            self._epochs_post.copy(),
            self._cfg,
            self._opts,
            self._forward,
        )
        self._calibration_params = calibration_params

        # Build list of ProcessedTrial from concatenated epochs
        pre_data = pre_epochs.get_data(copy=False)
        post_data = post_epochs.get_data(copy=False)
        trials = []
        for i in range(n_successful_trials):
            ep_pre = self._crop_pre_to_model_window(mne.EpochsArray(
                pre_data[i:i+1], info=pre_epochs.info,
                events=pre_epochs.events[i:i+1], tmin=pre_epochs.tmin, verbose=False,
            ))
            ep_post = self._crop_post_to_tep_window(mne.EpochsArray(
                post_data[i:i+1], info=post_epochs.info,
                events=post_epochs.events[i:i+1], tmin=post_epochs.tmin, verbose=False,
            ))
            trials.append(ProcessedTrial(ep_pre, ep_post))
        return trials

    @classmethod
    def from_bundle(cls, calibration_params, forward_path, source_sfreq: float, info: mne.Info):
        """Create a calibrated Preprocessor from pre-computed params (e.g. loaded from disk)."""
        instance = cls(forward_path, source_sfreq, info)
        instance._calibration_params = calibration_params
        return instance

    def preprocess(self, raw_pre: np.ndarray, raw_post: np.ndarray):
        """Resample and preprocess a single trial (both pre and post).

        Must be called after ``calibrate()``.

        Parameters
        ----------
        raw_pre : np.ndarray, shape (n_samples, n_channels)
            Raw pre-stimulus epoch at ``source_sfreq``.
        raw_post : np.ndarray, shape (n_samples, n_channels)
            Raw post-stimulus epoch at ``source_sfreq``.

        Returns
        -------
        ProcessedTrial or None
            Preprocessed trial with ``.epoch_pre`` and ``.epoch_post``,
            or ``None`` if either segment was rejected.
        """
        if self._calibration_params is None:
            raise RuntimeError("calibrate() must be called before preprocessing trials.")

        self._validate_raw_pre_shape(raw_pre)
        self._validate_raw_post_shape(raw_post)

        epoch_pre = _numpy_to_epochs_array(raw_pre, self._info, self._raw_pre_epoch_tmin)
        epoch_pre = epoch_pre.copy().crop(self._pre_stim_tmin, self._pre_stim_tmax)
        epoch_pre.resample(self._cfg.target_sfreq, method='polyphase')
        result_pre = preprocess_pre_trial(epoch_pre, self._calibration_params, self._cfg)
        if result_pre is False:
            return None
        result_pre = self._crop_pre_to_model_window(result_pre)

        epoch_post = _numpy_to_epochs_array(raw_post, self._info, self._raw_post_epoch_tmin)
        epoch_post.resample(self._cfg.target_sfreq, method='polyphase')
        result_post = preprocess_post_trial(epoch_post, self._calibration_params, self._cfg)
        if result_post is False:
            return None
        result_post = self._crop_post_to_tep_window(result_post)

        return ProcessedTrial(result_pre, result_post)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def calibration_params(self):
        """Calibration parameters dict (available after ``calibrate()`` is called)."""
        return self._calibration_params

    @property
    def n_trials(self):
        """Number of trials accumulated so far."""
        if self._epochs_pre is None:
            return 0
        return len(self._epochs_pre)
