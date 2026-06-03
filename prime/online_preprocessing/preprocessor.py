"""
Preprocessor: stateful object that accumulates trials, runs calibration, and
preprocesses individual trials using the resulting calibration parameters.

Usage
-----
    preprocessor = Preprocessor(forward_path)
    preprocessor.add_raw_pre(raw_pre)   # (n_samples, n_channels)
    preprocessor.add_raw_post(raw_post)

    cal_pre, cal_post = preprocessor.calibrate()

    epoch_pre = preprocessor.preprocess_pre(raw_pre)   # None if rejected
    epoch_post = preprocessor.preprocess_post(raw_post)
"""

import warnings
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from fractions import Fraction

import mne
import numpy as np
from scipy.stats import median_abs_deviation, zscore
from scipy.signal import butter, filtfilt, resample_poly


from prime.prime_config import (
    epoch_n_times,
    get_dipole_time_range,
    get_model_time_range,
    get_processed_sfreq,
    get_post_time_range,
    get_calibration_time_range,
    get_raw_sfreq,
    time_to_sample,
)
from prime.online_preprocessing.utils.ica_calibrator import get_number_of_components, get_ica
from prime.online_preprocessing.utils.ssp_sir_python import (
    ssp_sir_to_average,
    ssp_sir_trials,
    ssp_sir_single_trial,
)
from prime.online_preprocessing.utils.sound_modified import sound
from prime.online_preprocessing.utils.channel_interpolations import (
    custom_get_interpolation_matrix,
    apply_channel_interpolation,
)
from prime.online_preprocessing.config import get_default_config

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


@contextmanager
def _profile(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[profile] {label}: {elapsed * 1000:.1f}ms")


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
    model_tmin: float,
    model_tmax: float,
    calibration_tmin: float,
    calibration_tmax: float,
) -> None:
    _validate_time_range_within(
        "Pre epoch", model_tmin, model_tmax, calibration_tmin, calibration_tmax,
    )


def _require_info_sfreq(info: mne.Info, sfreq: float) -> None:
    if abs(info["sfreq"] - sfreq) > 1e-6:
        raise ValueError(
            f"info['sfreq'] must be {sfreq} (configs/prime.yaml), got {info['sfreq']}"
        )


def _mne_info_from_forward(forward: mne.Forward) -> mne.Info:
    raw_sfreq = get_raw_sfreq()
    montage = mne.channels.make_standard_montage('standard_1005')
    info = mne.create_info(
        ch_names=forward.ch_names,
        sfreq=raw_sfreq,
        ch_types='eeg',
    )
    info.set_montage(montage)
    _require_info_sfreq(info, raw_sfreq)
    return info


def _validate_post_epoch_window(
    dipole_tmin: float,
    dipole_tmax: float,
    post_tmin: float,
    post_tmax: float,
) -> None:
    if not (
        post_tmin <= dipole_tmin
        and dipole_tmax <= post_tmax
        and dipole_tmin < dipole_tmax
    ):
        raise ValueError(
            f"Post epoch window [{dipole_tmin}, {dipole_tmax}] must satisfy "
            f"{post_tmin} <= dipole_tmin < dipole_tmax <= {post_tmax}"
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
    """Crop (n_samples, n_channels) to [tmin, tmax] inclusive relative to the event.

    Matches ``epoch_n_times`` / MNE ``crop(..., include_tmax=True)`` on an epoch whose
    first sample is at ``time_offsets[0]``, using ``raw_sfreq`` from configs/prime.yaml.
    """
    time_offsets = np.asarray(time_offsets, dtype=np.float64)
    if time_offsets.size < 1:
        raise ValueError("time_offsets must not be empty")
    sfreq = get_raw_sfreq()
    start = time_to_sample(tmin, float(time_offsets[0]), sfreq)
    stop = start + epoch_n_times(tmin, tmax, sfreq)
    if start < 0 or stop > eeg_buffer.shape[0]:
        raise ValueError(
            f"cannot crop [{tmin}, {tmax}] from buffer "
            f"(start={start}, stop={stop}, buffer_len={eeg_buffer.shape[0]})"
        )
    return eeg_buffer[start:stop]


def _resample_buffer_polyphase(
    data: np.ndarray,
    sfreq_from: float,
    sfreq_to: float,
) -> np.ndarray:
    """Resample (n_samples, n_channels) using scipy polyphase."""
    if abs(sfreq_from - sfreq_to) < 1e-9:
        return data
    ratio = Fraction(str(sfreq_to)) / Fraction(str(sfreq_from))
    return resample_poly(
        data,
        up=ratio.numerator,
        down=ratio.denominator,
        axis=0,
    )


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
    info_processed: mne.Info,
    cfg,
    calibration_tmin: float,
    calibration_tmax: float,
    post_tmin: float,
    post_tmax: float,
):
    """Append one raw pre/post pair (resampled) to calibration epoch structs."""
    raw_pre_time_offsets = np.array([calibration_tmin], dtype=np.float64)
    raw_post_time_offsets = np.array([post_tmin], dtype=np.float64)
    processed_sfreq = float(info_processed["sfreq"])
    raw_sfreq = float(info["sfreq"])

    pre_buf = crop_eeg_buffer(
        raw_pre,
        raw_pre_time_offsets,
        cfg.pre_stim_timerange[0],
        cfg.pre_stim_timerange[1],
    )
    pre_ica_buf = crop_eeg_buffer(
        raw_pre,
        raw_pre_time_offsets,
        cfg.ica_opts.pre_timerange[0],
        cfg.ica_opts.pre_timerange[1],
    )
    post_buf = crop_eeg_buffer(
        raw_post,
        raw_post_time_offsets,
        post_tmin,
        post_tmax,
    )

    pre_buf = _resample_buffer_polyphase(pre_buf, sfreq_from=raw_sfreq, sfreq_to=processed_sfreq)
    pre_ica_buf = _resample_buffer_polyphase(pre_ica_buf, sfreq_from=raw_sfreq, sfreq_to=processed_sfreq)
    post_buf = _resample_buffer_polyphase(post_buf, sfreq_from=raw_sfreq, sfreq_to=processed_sfreq)

    trial_pre = _numpy_to_epochs_array(pre_buf, info_processed, cfg.pre_stim_timerange[0])
    trial_pre_ica = _numpy_to_epochs_array(pre_ica_buf, info_processed, cfg.ica_opts.pre_timerange[0])
    trial_post = _numpy_to_epochs_array(post_buf, info_processed, post_tmin)
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

    model_tmin, model_tmax = get_model_time_range()
    dipole_tmin, dipole_tmax = get_dipole_time_range()

    epochs_pre = epochs_pre.crop(model_tmin, model_tmax, include_tmax=True)
    epochs_post = epochs_post.crop(dipole_tmin, dipole_tmax, include_tmax=True)

    calibration_params['ica'] = ica
    return (
        calibration_params,
        n_successful_trials,
        epochs_pre.get_data(copy=False),
        epochs_post.get_data(copy=False),
    )


# ==================== Preprocessor class ====================

class Preprocessor:
    """Accumulates raw trials and produces calibration parameters on demand."""

    def __init__(self, forward_path):
        cfg = get_default_config()
        raw_sfreq = get_raw_sfreq()
        self._forward = mne.read_forward_solution(str(forward_path), verbose=False)
        info = _mne_info_from_forward(self._forward)
        calibration_tmin, calibration_tmax = get_calibration_time_range()
        post_tmin, post_tmax = get_post_time_range()
        model_tmin, model_tmax = get_model_time_range()
        dipole_tmin, dipole_tmax = get_dipole_time_range()
        _validate_time_range_within(
            "Pre-stim preprocessing",
            cfg.pre_stim_timerange[0],
            cfg.pre_stim_timerange[1],
            calibration_tmin,
            calibration_tmax,
        )
        _validate_time_range_within(
            "ICA",
            cfg.ica_opts.pre_timerange[0],
            cfg.ica_opts.pre_timerange[1],
            calibration_tmin,
            calibration_tmax,
        )
        _validate_pre_epoch_window(
            model_tmin, model_tmax, calibration_tmin, calibration_tmax,
        )
        _validate_post_epoch_window(
            dipole_tmin, dipole_tmax, post_tmin, post_tmax,
        )
        self._cfg = cfg
        self._info = info
        self._calibration_tmin = calibration_tmin
        self._calibration_tmax = calibration_tmax
        self._post_tmin = post_tmin
        self._post_tmax = post_tmax
        self._raw_pre_n_times = epoch_n_times(
            calibration_tmin, calibration_tmax, raw_sfreq,
        )
        self._raw_post_n_times = epoch_n_times(
            post_tmin, post_tmax, raw_sfreq,
        )
        self._pre_stim_tmin = cfg.pre_stim_timerange[0]
        self._pre_stim_tmax = cfg.pre_stim_timerange[1]
        self._model_tmin = model_tmin
        self._model_tmax = model_tmax
        self._dipole_tmin = dipole_tmin
        self._dipole_tmax = dipole_tmax
        self._opts = cfg.to_dicts()
        self._processed_sfreq = get_processed_sfreq()
        self._raw_pre_time_offsets = np.array([calibration_tmin], dtype=np.float64)
        self._raw_post_time_offsets = np.array([post_tmin], dtype=np.float64)
        self._info_processed = self._info.copy()
        with self._info_processed._unlock():
            self._info_processed["sfreq"] = self._processed_sfreq

        self._epochs_pre = None
        self._epochs_pre_ica = None
        self._epochs_post = None
        self._calibration_params = None
        self._pending_pre = None
        self._awaiting_post = False

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
        return epoch_pre.crop(self._model_tmin, self._model_tmax, include_tmax=True)

    def _crop_post_to_tep_window(self, epoch_post: mne.Epochs) -> mne.Epochs:
        return epoch_post.crop(self._dipole_tmin, self._dipole_tmax, include_tmax=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_raw_pre(self, raw_pre: np.ndarray) -> None:
        """Buffer one raw pre-stimulus epoch for calibration.

        Parameters
        ----------
        raw_pre : np.ndarray, shape (n_samples, n_channels)
            EEG for ``[calibration_tmin, calibration_tmax]`` at ``raw_sfreq``.
        """
        if self._awaiting_post:
            raise RuntimeError(
                "add_raw_post must be called before the next add_raw_pre"
            )
        self._validate_raw_pre_shape(raw_pre)
        self._pending_pre = np.asarray(raw_pre, dtype=np.float64)
        self._awaiting_post = True

    def add_raw_post(self, raw_post: np.ndarray) -> None:
        """Append the paired pre/post epochs to the calibration buffers.

        Parameters
        ----------
        raw_post : np.ndarray, shape (n_samples, n_channels)
            EEG for ``[post_tmin, post_tmax]`` at ``raw_sfreq``.
        """
        if not self._awaiting_post:
            raise RuntimeError("add_raw_pre must be called before add_raw_post")
        self._validate_raw_post_shape(raw_post)
        self._epochs_pre, self._epochs_pre_ica, self._epochs_post = append_calibration_epochs(
            self._epochs_pre, self._epochs_pre_ica, self._epochs_post,
            self._pending_pre, raw_post,
            self._info, self._info_processed, self._cfg,
            self._calibration_tmin, self._calibration_tmax,
            self._post_tmin, self._post_tmax,
        )
        self._pending_pre = None
        self._awaiting_post = False

    def calibrate(self):
        """Run calibration on the accumulated trials.

        Stores the resulting parameters internally.  Call ``preprocess_pre`` /
        ``preprocess_post`` on individual trials afterwards.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Pre- and post-stimulus calibration epochs that survived rejection.
        """
        if self._awaiting_post:
            raise RuntimeError("add_raw_post was not called for the last pre epoch")
        if self._epochs_pre is None:
            raise RuntimeError("No trials have been added yet.")

        calibration_params, n_successful_trials, pre_data, post_data = preprocess_calibration(
            self._epochs_pre.copy(),
            self._epochs_pre_ica.copy(),
            self._epochs_post.copy(),
            self._cfg,
            self._opts,
            self._forward,
        )
        self._calibration_params = calibration_params

        if n_successful_trials != pre_data.shape[0] or n_successful_trials != post_data.shape[0]:
            raise RuntimeError(
                "mismatch between successful trial count and calibration epoch arrays"
            )

        return pre_data, post_data

    @classmethod
    def from_bundle(cls, calibration_params, forward_path):
        """Create a calibrated Preprocessor from pre-computed params (e.g. loaded from disk)."""
        instance = cls(forward_path)
        instance._calibration_params = calibration_params
        return instance

    def preprocess_pre(
        self,
        raw_pre: np.ndarray,
    ) -> np.ndarray | None:
        """Resample and preprocess a single pre-stimulus trial.

        Must be called after ``calibrate()``.

        Parameters
        ----------
        raw_pre : np.ndarray
            Raw pre-stimulus trial with shape (n_samples, n_channels).

        Returns a NumPy array with shape (n_channels, n_times), or ``None`` if
        the trial was rejected.
        """
        if self._calibration_params is None:
            raise RuntimeError("calibrate() must be called before preprocessing trials.")

        self._validate_raw_pre_shape(raw_pre)
        with _profile("preprocess_pre: crop_eeg_buffer"):
            pre_stim = crop_eeg_buffer(
                raw_pre,
                self._raw_pre_time_offsets,
                self._pre_stim_tmin,
                self._pre_stim_tmax,
            )
        with _profile("preprocess_pre: resample_polyphase"):
            pre_stim = _resample_buffer_polyphase(
                pre_stim,
                sfreq_from=self._info["sfreq"],
                sfreq_to=self._processed_sfreq,
            )
        # Keep pre-stim single-trial preprocessing fully in NumPy.
        with _profile("preprocess_pre: to_numpy_float64"):
            data = pre_stim.T[np.newaxis, :, :].astype(np.float64, copy=False)

        if self._calibration_params['bad_channels']:
            with _profile("preprocess_pre: interpolate_bad_channels"):
                interpolation_info = self._calibration_params['channel_interpolation_info']
                data[..., interpolation_info['bads_idx'], :] = np.matmul(
                    interpolation_info['interpolation_matrix'],
                    data[..., interpolation_info['goods_idx'], :],
                )

        filter_opts = self._opts['filter_opts']
        with _profile("preprocess_pre: apply_filter"):
            data = _apply_filter(
                data,
                self._calibration_params['pre_stim_filter'],
                filter_opts['pad_time'],
                self._processed_sfreq,
            )

        # Mean subtraction (average reference).
        with _profile("preprocess_pre: mean_subtraction"):
            data -= np.mean(data, axis=1, keepdims=True)

        trial_reject_opts = self._opts['trial_reject_opts']

        # Global MAD check.
        with _profile("preprocess_pre: global_mad_check"):
            mad_val = median_abs_deviation(data, axis=(1, 2))[0]
            z_mad = (
                mad_val - self._calibration_params['good_trial_stats_pre']['mads_mean']
            ) / self._calibration_params['good_trial_stats_pre']['mads_std']
            threshold = trial_reject_opts['pre']['global_zscore_threshold']
            if z_mad < threshold[0] or z_mad > threshold[1]:
                return None

        # Local MAD check.
        with _profile("preprocess_pre: local_mad_check"):
            local_mad = median_abs_deviation(data, axis=2)
            z_local = zscore(local_mad, axis=1)
            if np.any(np.abs(z_local) > trial_reject_opts['pre']['local_zscore_threshold']):
                return None

        with _profile("preprocess_pre: crop_to_model_window"):
            start = time_to_sample(
                self._model_tmin,
                self._pre_stim_tmin,
                self._processed_sfreq,
            )
            stop = start + epoch_n_times(
                self._model_tmin,
                self._model_tmax,
                self._processed_sfreq,
            )
            if start < 0 or stop > data.shape[2]:
                raise ValueError(
                    f"cannot crop pre window [{self._model_tmin}, {self._model_tmax}] "
                    f"from pre-stim data (start={start}, stop={stop}, n_times={data.shape[2]})"
                )
            # Slicing can produce non-contiguous views; force contiguous layout
            # so torch.from_numpy() never sees negative/unsupported strides.
            model_window = data[0, :, start:stop]
            return np.ascontiguousarray(model_window, dtype=np.float64)

    def preprocess_post(self, raw_post: np.ndarray) -> np.ndarray | None:
        """Resample and preprocess a single post-stimulus trial.

        Must be called after ``calibrate()``.

        Returns a NumPy array with shape (1, n_channels, n_samples), or ``None`` if
        the trial was rejected.
        """
        if self._calibration_params is None:
            raise RuntimeError("calibrate() must be called before preprocessing trials.")

        self._validate_raw_post_shape(raw_post)
        post_buf = crop_eeg_buffer(
            raw_post,
            self._raw_post_time_offsets,
            self._post_tmin,
            self._post_tmax,
        )
        post_buf = _resample_buffer_polyphase(
            post_buf,
            sfreq_from=self._info["sfreq"],
            sfreq_to=self._processed_sfreq,
        )
        epoch_post = _numpy_to_epochs_array(post_buf, self._info_processed, self._post_tmin)
        calibration_params = self._calibration_params
        dicts = self._cfg.to_dicts()
        trial_reject_opts = dicts["trial_reject_opts"]

        epoch_post.apply_baseline(self._cfg.baseline)
        epoch_post = mne.preprocessing.fix_stim_artifact(
            epoch_post,
            tmin=self._cfg.artifact_window_1[0],
            tmax=self._cfg.artifact_window_1[1],
            mode="window",
        )

        if calibration_params["bad_channels"]:
            epoch_post, _ = _interpolate_bad_channels(
                epoch_post,
                calibration_params["bad_channels"],
                calibration_params["channel_interpolation_info"],
            )

        epoch_post.set_eeg_reference("average", projection=False, verbose=False)

        ica = calibration_params["ica"]

        # Check ocular ICA components
        source_time_course = ica.get_sources(epoch_post)
        source_data = source_time_course.get_data(copy=True)
        for component_idx in calibration_params["ocular_thresholds_post"]:
            component_info = calibration_params["ocular_thresholds_post"][component_idx]
            z_comp = (
                np.abs(
                    source_data[
                        0,
                        component_idx,
                        component_info["time_indices_of_interest"],
                    ]
                )
                - component_info["mean"]
            ) / component_info["std"]
            if np.median(z_comp) > trial_reject_opts["ocular"]["z_threshold"]:
                return None

        ica.apply(epoch_post)

        epoch_post.apply_baseline(self._cfg.baseline).set_eeg_reference(
            "average", projection=False, verbose=False
        )
        epoch_post.crop(calibration_params["post_mintime"], None)

        # Apply SOUND
        data = epoch_post.get_data(copy=True)
        data = np.matmul(calibration_params["sound_filter"], data)
        epoch_post = mne.EpochsArray(
            data,
            epoch_post.info,
            events=epoch_post.events,
            tmin=epoch_post.times[0],
            verbose=False,
        )
        epoch_post.set_eeg_reference("average", projection=False, verbose=False)

        # Apply SSP-SIR
        data = epoch_post.get_data(copy=True)
        data = ssp_sir_single_trial(
            data[0, :, :],
            calibration_params["sspsir_suppression_matrix_P"],
            calibration_params["sspsir_sir_projmat_suppr"],
            calibration_params["sspsir_sir_projmat_orig"],
            calibration_params["sspsir_filter_kernel"],
        )
        epoch_post = mne.EpochsArray(
            data.reshape(1, data.shape[0], data.shape[1]),
            epoch_post.info,
            events=epoch_post.events,
            tmin=epoch_post.times[0],
        )

        epoch_post = mne.preprocessing.fix_stim_artifact(
            epoch_post,
            tmin=calibration_params["post_mintime"],
            tmax=self._cfg.artifact_window_2[1],
            mode="window",
        )
        epoch_post.set_eeg_reference("average", projection=False, verbose=False)

        # Global MAD check
        reject_data = epoch_post.copy().crop(
            self._cfg.reject_range[0], self._cfg.reject_range[1]
        ).get_data(copy=True)
        mad_val = median_abs_deviation(reject_data, axis=(1, 2))[0]
        z_mad = (
            mad_val - calibration_params["good_trial_stats_post"]["mads_mean"]
        ) / calibration_params["good_trial_stats_post"]["mads_std"]
        threshold = trial_reject_opts["post"]["global_zscore_threshold"]
        if z_mad < threshold[0] or z_mad > threshold[1]:
            return None

        # Local MAD check
        local_mad = median_abs_deviation(reject_data, axis=2)
        z_local = zscore(local_mad, axis=1)
        if np.any(np.abs(z_local) > trial_reject_opts["post"]["local_zscore_threshold"]):
            return None

        return self._crop_post_to_tep_window(epoch_post).get_data(copy=False)

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
