"""
Preprocessor: stateful object that accumulates trials, runs calibration, and
preprocesses individual trials using the resulting calibration parameters.

Usage
-----
    preprocessor = Preprocessor(forward_path)
    preprocessor.add_trial(eeg_buffer, relative_timestamps)  # (n_samples, n_channels)

    model_buffers, dipole_buffers = preprocessor.calibrate()

    epoch_pre = preprocessor.preprocess_pre(eeg_buffer, timestamps)   # None if rejected
    epoch_post = preprocessor.preprocess_post(eeg_buffer, relative_timestamps)
"""

import warnings
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import mne
import numpy as np
from scipy.stats import median_abs_deviation, zscore
from scipy.signal import butter, filtfilt


from prime_core.prime_config import (
    epoch_n_times,
    get_dipole_time_range,
    get_qc_time_range,
    get_ica_time_range,
    get_model_time_range,
    get_processed_sfreq,
    get_post_initial_time_range,
    get_post_time_range,
    get_raw_sfreq,
    time_to_sample,
)
from prime_core.preprocessing.utils.ica_calibrator import get_number_of_components, get_ica
from prime_core.preprocessing.utils.ssp_sir_python import (
    ssp_sir_to_average,
    ssp_sir_trials,
    ssp_sir_single_trial,
)
from prime_core.preprocessing.utils.sound_modified import sound
from prime_core.preprocessing.utils.channel_interpolations import (
    custom_get_interpolation_matrix,
    apply_channel_interpolation,
)
from prime_core.preprocessing.utils.mad import (
    SingleTrialMadWorkspace,
    global_mad_zscore_rejected,
    local_mad_zscore_rejected,
)
from prime_core.preprocessing.utils.resampling import resample_buffer_polyphase
from prime_core.preprocessing.config import get_default_config

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "offline_data"


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
    filtered = filtfilt(coeffs[0], coeffs[1], padded, padlen=0)
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


def _detect_bad_channels_qc(epochs, options, filter_opts):
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
    *,
    sfreq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop (n_samples, n_channels) to [tmin, tmax] inclusive relative to the event.

    Matches ``epoch_n_times`` / MNE ``crop(..., include_tmax=True)`` on an epoch whose
    first sample is at ``time_offsets[0]``. Uses ``raw_sfreq`` from configs/prime.yaml
    unless ``sfreq`` is given (e.g. after resampling to ``processed_sfreq``).

    Returns
    -------
    cropped_buffer : np.ndarray
        EEG samples in the requested window.
    cropped_time_offsets : np.ndarray
        Time offset of each cropped sample, sliced from ``time_offsets``.
    """
    time_offsets = np.asarray(time_offsets, dtype=np.float64)
    if time_offsets.size < 1:
        raise ValueError("time_offsets must not be empty")
    if sfreq is None:
        sfreq = get_raw_sfreq()
    start = time_to_sample(tmin, float(time_offsets[0]), sfreq)
    stop = start + epoch_n_times(tmin, tmax, sfreq)
    if start < 0 or stop > eeg_buffer.shape[0]:
        print(
            f"crop_eeg_buffer: cannot crop [{tmin}, {tmax}] from buffer "
            f"(start={start}, stop={stop}, buffer_len={eeg_buffer.shape[0]})\n"
            f"  time_offsets: first={time_offsets[0]:.6f}, last={time_offsets[-1]:.6f}, "
            f"n={len(time_offsets)}, sfreq={sfreq}"
        )
        raise ValueError(
            f"cannot crop [{tmin}, {tmax}] from buffer "
            f"(start={start}, stop={stop}, buffer_len={eeg_buffer.shape[0]})"
        )
    return eeg_buffer[start:stop], time_offsets[start:stop]


def crop_mne_trial_to_raw_epochs(
    trial: mne.BaseEpochs,
    raw_pre_tmin: float,
    raw_pre_tmax: float,
    raw_post_initial_tmin: float,
    raw_post_initial_tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a full-trial EpochsArray into raw pre/post numpy buffers."""
    raw_pre = _epochs_to_buffer(
        trial.copy().crop(raw_pre_tmin, raw_pre_tmax, include_tmax=True),
    )
    raw_post = _epochs_to_buffer(
        trial.copy().crop(raw_post_initial_tmin, raw_post_initial_tmax, include_tmax=True),
    )
    return raw_pre, raw_post


def crop_mne_trial_to_buffer(
    trial: mne.BaseEpochs,
    trial_tmin: float,
    trial_tmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a full-trial EpochsArray to the combined pre/post window."""
    cropped = trial.copy().crop(trial_tmin, trial_tmax, include_tmax=True)
    return _epochs_to_buffer(cropped), cropped.times.astype(np.float64)


# ==================== Calibration pipeline ====================


def preprocess_calibration(qc_epochs, ica_epochs, post_epochs, cfg, opts, forward):
    """Full calibration preprocessing pipeline for qc, ica, and post epochs.

    Returns ``(calibration_params, n_successful_trials, model_buffers, dipole_buffers)``
    where the last two arrays are cropped to the model and dipole time windows.
    """
    channel_reject_opts = opts['channel_reject_opts']
    ica_opts = opts['ica_opts']
    trial_reject_opts = opts['trial_reject_opts']
    sound_opts = opts['sound_opts']
    ssp_sir_opts = opts['ssp_sir_opts']
    filter_opts = opts['filter_opts']

    leadfield = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0)
    calibration_params = {}

    post_epochs.apply_baseline(cfg.baseline)
    post_epochs = mne.preprocessing.fix_stim_artifact(
        post_epochs,
        tmin=cfg.artifact_window_1[0],
        tmax=cfg.artifact_window_1[1],
        mode='window',
    )

    bad_channels_qc, qc_filter = _detect_bad_channels_qc(
        qc_epochs, channel_reject_opts['pre'], filter_opts,
    )
    bad_channels_post = _detect_bad_channels_post(
        post_epochs, channel_reject_opts['post'], cfg.reject_range,
    )
    bad_channels = list(np.union1d(bad_channels_qc, bad_channels_post))

    if bad_channels:
        qc_epochs, interpolation_info = _interpolate_bad_channels(qc_epochs, bad_channels, None)
        ica_epochs, _ = _interpolate_bad_channels(ica_epochs, bad_channels, interpolation_info)
        post_epochs, _ = _interpolate_bad_channels(post_epochs, bad_channels, interpolation_info)
    else:
        interpolation_info = None

    calibration_params['qc_filter'] = qc_filter
    calibration_params['bad_channels'] = bad_channels
    calibration_params['channel_interpolation_info'] = interpolation_info

    ica_data = ica_epochs.get_data(copy=True)
    ica_data, _ = _butter_filter(
        ica_data,
        ica_opts['filtering']['cutoff'],
        'bandpass',
        ica_epochs.info['sfreq'],
        ica_opts['filtering']['order_bandpass'],
        ica_opts['filtering']['pad_time_bandpass'],
    )
    ica_epochs = mne.EpochsArray(
        ica_data, info=ica_epochs.info, events=ica_epochs.events, tmin=ica_epochs.times[0],
    )
    post_epochs.set_eeg_reference('average', projection=False, verbose=False)
    ica_epochs.set_eeg_reference('average', projection=False, verbose=False)

    ica, excluded_components, _ = get_ica(
        ica_epochs,
        ica_opts['pc_threshold'],
        None,
        ica_opts['bad_component_thresholds'],
        ica_opts['n_min_comps_to_reject'],
        ica_opts['threshold_min_components_to_reject'],
    )
    del ica_epochs

    qc_data = qc_epochs.get_data(copy=True)
    qc_data = _apply_filter(qc_data, qc_filter, filter_opts['pad_time'], qc_epochs.info['sfreq'])
    qc_epochs = mne.EpochsArray(
        qc_data, info=qc_epochs.info, events=qc_epochs.events, tmin=qc_epochs.times[0],
    )
    del qc_data

    bad_ocular_post, ocular_threshold_post = _detect_ocular_trials(
        ica, post_epochs, trial_reject_opts['ocular']['post_timerange'][0],
        trial_reject_opts['ocular']['post_timerange'][1],
        excluded_components['eye blink'], trial_reject_opts['ocular']['z_threshold'],
    )

    calibration_params['ocular_threshold_post'] = ocular_threshold_post

    ica.apply(post_epochs)

    qc_epochs, post_epochs = _drop_bad_trials([qc_epochs, post_epochs], bad_ocular_post)
    qc_epochs.set_eeg_reference('average', projection=False, verbose=False)

    bad_qc, stats_qc = _find_bad_trials(
        qc_epochs,
        trial_reject_opts['pre']['global_zscore_threshold'],
        trial_reject_opts['pre']['local_zscore_threshold'],
        False,
        False,
    )
    calibration_params['good_trial_stats_qc'] = stats_qc
    qc_epochs, post_epochs = _drop_bad_trials([qc_epochs, post_epochs], bad_qc)

    post_epochs.apply_baseline(cfg.baseline).set_eeg_reference('average', projection=False, verbose=False)
    post_tmin, post_tmax = get_post_time_range()
    post_epochs.crop(post_tmin, post_tmax, include_tmax=True)

    post_data = post_epochs.get_data(copy=True)
    evoked = np.mean(post_data, axis=0)
    n_channels = evoked.shape[0]
    sound_filter, _, _, _ = sound(
        evoked.T, 0, np.ones((n_channels, 1)), n_channels, leadfield,
        sound_opts['max_iterations'], sound_opts['lambda'],
        sound_opts['convergence_tolerance'], sound_opts['fixed_max_iterations'],
    )

    for i in range(post_data.shape[0]):
        post_data[i, :, :] = np.matmul(sound_filter, post_data[i, :, :])

    post_epochs = mne.EpochsArray(
        post_data, post_epochs.info, events=post_epochs.events, tmin=post_epochs.times[0],
    )
    post_epochs.set_eeg_reference('average', projection=False, verbose=False)
    calibration_params['sound_filter'] = sound_filter

    post_data = post_epochs.get_data(copy=True)
    evoked = np.mean(post_data, axis=0)
    _, _, _, kernel, P, _, _, projection_suppression, projection_original, _ = ssp_sir_to_average(
        evoked, leadfield, post_epochs.info['sfreq'], ssp_sir_opts['timerange'], method=ssp_sir_opts['method'],
    )

    post_data = ssp_sir_trials(post_data, P, projection_suppression, projection_original, kernel)
    post_epochs = mne.EpochsArray(
        post_data, post_epochs.info, events=post_epochs.events, tmin=post_epochs.times[0],
    )

    calibration_params['sspsir_suppression_matrix_P'] = P
    calibration_params['sspsir_filter_kernel'] = kernel
    calibration_params['sspsir_sir_projmat_suppr'] = projection_suppression
    calibration_params['sspsir_sir_projmat_orig'] = projection_original

    post_epochs = mne.preprocessing.fix_stim_artifact(
        post_epochs, tmin=post_tmin, tmax=cfg.artifact_window_2[1], mode='window',
    )
    post_epochs.set_eeg_reference('average', projection=False, verbose=False)

    bad_post, stats_post = _find_bad_trials(
        post_epochs.copy().crop(cfg.reject_range[0], cfg.reject_range[1]),
        trial_reject_opts['post']['global_zscore_threshold'],
        trial_reject_opts['post']['local_zscore_threshold'],
        False,
        False,
    )
    calibration_params['good_trial_stats_post'] = stats_post
    qc_epochs, post_epochs = _drop_bad_trials([qc_epochs, post_epochs], bad_post)

    n_successful_trials = qc_epochs.get_data(copy=True).shape[0]

    model_tmin, model_tmax = get_model_time_range()
    dipole_tmin, dipole_tmax = get_dipole_time_range()

    model_epochs = qc_epochs.crop(model_tmin, model_tmax, include_tmax=True)
    dipole_epochs = post_epochs.crop(dipole_tmin, dipole_tmax, include_tmax=True)

    calibration_params['ica'] = ica
    model_buffers = model_epochs.get_data(copy=False)
    dipole_buffers = dipole_epochs.get_data(copy=False)
    return calibration_params, n_successful_trials, model_buffers, dipole_buffers


# ==================== Preprocessor class ====================

class Preprocessor:
    """Accumulates raw trials and produces calibration parameters on demand."""

    def __init__(self, forward_path):
        cfg = get_default_config()
        raw_sfreq = get_raw_sfreq()
        self._forward = mne.read_forward_solution(str(forward_path), verbose=False)
        info = _mne_info_from_forward(self._forward)
        ica_tmin, ica_tmax = get_ica_time_range()
        post_initial_tmin, post_initial_tmax = get_post_initial_time_range()
        post_tmin, post_tmax = get_post_time_range()
        qc_tmin, qc_tmax = get_qc_time_range()
        model_tmin, model_tmax = get_model_time_range()
        dipole_tmin, dipole_tmax = get_dipole_time_range()

        self._cfg = cfg
        self._info = info
        self._ica_tmin = ica_tmin
        self._ica_tmax = ica_tmax
        self._post_initial_tmin = post_initial_tmin
        self._post_initial_tmax = post_initial_tmax
        self._post_tmin = post_tmin
        self._post_tmax = post_tmax
        self._qc_tmin = qc_tmin
        self._qc_tmax = qc_tmax
        self._model_tmin = model_tmin
        self._model_tmax = model_tmax
        self._dipole_tmin = dipole_tmin
        self._dipole_tmax = dipole_tmax
        self._opts = cfg.to_dicts()
        self._processed_sfreq = get_processed_sfreq()
        self._info_processed = self._info.copy()
        with self._info_processed._unlock():
            self._info_processed["sfreq"] = self._processed_sfreq

        self.qc_epochs = None
        self.ica_epochs = None
        self.post_epochs = None
        self._calibration_params = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_trial(
        self,
        eeg_buffer: np.ndarray,
        relative_timestamps: np.ndarray,
    ) -> None:
        """Append one raw trial to the calibration buffers.

        Parameters
        ----------
        eeg_buffer : np.ndarray, shape (n_samples, n_channels)
            EEG covering ``[trial_tmin, trial_tmax]`` at ``raw_sfreq``.
        relative_timestamps : np.ndarray, shape (n_samples,)
            Time of each sample in seconds relative to the TMS event.
        """
        processed_sfreq = get_processed_sfreq()
        raw_sfreq = get_raw_sfreq()

        qc_buffer, _ = crop_eeg_buffer(
            eeg_buffer,
            relative_timestamps,
            self._qc_tmin,
            self._qc_tmax,
        )
        ica_buffer, _ = crop_eeg_buffer(
            eeg_buffer,
            relative_timestamps,
            self._ica_tmin,
            self._ica_tmax,
        )
        post_buffer, _ = crop_eeg_buffer(
            eeg_buffer,
            relative_timestamps,
            self._post_initial_tmin,
            self._post_initial_tmax,
        )

        qc_buffer = resample_buffer_polyphase(qc_buffer, sfreq_from=raw_sfreq, sfreq_to=processed_sfreq)
        ica_buffer = resample_buffer_polyphase(ica_buffer, sfreq_from=raw_sfreq, sfreq_to=processed_sfreq)
        post_buffer = resample_buffer_polyphase(post_buffer, sfreq_from=raw_sfreq, sfreq_to=processed_sfreq)

        qc_trial = _numpy_to_epochs_array(qc_buffer, self._info_processed, self._qc_tmin)
        ica_trial = _numpy_to_epochs_array(ica_buffer, self._info_processed, self._ica_tmin)
        post_trial = _numpy_to_epochs_array(post_buffer, self._info_processed, self._post_initial_tmin)

        if self.qc_epochs is None:
            self.qc_epochs = qc_trial
            self.ica_epochs = ica_trial
            self.post_epochs = post_trial
        else:
            self.qc_epochs = mne.concatenate_epochs([self.qc_epochs, qc_trial])
            self.ica_epochs = mne.concatenate_epochs([self.ica_epochs, ica_trial])
            self.post_epochs = mne.concatenate_epochs([self.post_epochs, post_trial])

    def calibrate(self):
        """Run calibration on the accumulated trials.

        Stores the resulting parameters internally.  Call ``preprocess_pre`` /
        ``preprocess_post`` on individual trials afterwards.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(model_buffers, dipole_buffers)`` — preprocessed calibration trials cropped
            to the model and dipole time windows, respectively.
        """
        if self.qc_epochs is None:
            raise RuntimeError("No trials have been added yet.")

        calibration_params, n_successful_trials, model_buffers, dipole_buffers = (
            preprocess_calibration(
                self.qc_epochs.copy(),
                self.ica_epochs.copy(),
                self.post_epochs.copy(),
                self._cfg,
                self._opts,
                self._forward,
            )
        )
        self._calibration_params = calibration_params
        qc_shape = self.qc_epochs.get_data(copy=False).shape
        self._mad_workspace = SingleTrialMadWorkspace(qc_shape[1], qc_shape[2])

        if (
            n_successful_trials != model_buffers.shape[0]
            or n_successful_trials != dipole_buffers.shape[0]
        ):
            raise RuntimeError(
                "mismatch between successful trial count and calibration epoch arrays"
            )

        return model_buffers, dipole_buffers

    @classmethod
    def from_bundle(cls, calibration_params, forward_path):
        """Create a calibrated Preprocessor from pre-computed params (e.g. loaded from disk)."""
        instance = cls(forward_path)
        instance._calibration_params = calibration_params
        n_channels = len(instance._info["ch_names"])
        n_times = epoch_n_times(instance._qc_tmin, instance._qc_tmax, instance._processed_sfreq)
        instance._mad_workspace = SingleTrialMadWorkspace(n_channels, n_times)
        return instance

    def preprocess_pre(
        self,
        eeg_buffer: np.ndarray,
        timestamps: np.ndarray,
        *,
        from_pulse: bool = True,
    ) -> np.ndarray | None:
        """Resample and preprocess a single pre-stimulus trial.

        Must be called after ``calibrate()``.

        Parameters
        ----------
        eeg_buffer : np.ndarray, shape (n_samples, n_channels)
            EEG covering ``[trial_tmin, trial_tmax]`` at ``raw_sfreq``.
        timestamps : np.ndarray, shape (n_samples,)
            Time of each sample in seconds relative to the TMS pulse when
            ``from_pulse`` is true; relative to the periodic reference (ending
            at 0 s) when false.
        from_pulse : bool, default True
            When false (periodic call), timestamps are shifted by ``qc_tmax``
            to align them with the pulse before cropping.

        Returns a NumPy array with shape (n_channels, n_times), or ``None`` if
        the trial was rejected.
        """
        if self._calibration_params is None:
            raise RuntimeError("calibrate() must be called before preprocessing trials.")

        timestamps = np.asarray(timestamps, dtype=np.float64)
        if not from_pulse:
            timestamps = timestamps + self._qc_tmax

        pre_stim, _ = crop_eeg_buffer(
            eeg_buffer,
            timestamps,
            self._qc_tmin,
            self._qc_tmax,
        )
        pre_stim = resample_buffer_polyphase(
            pre_stim,
            sfreq_from=self._info["sfreq"],
            sfreq_to=self._processed_sfreq,
        )

        # Keep pre-stim single-trial preprocessing fully in NumPy.
        data = pre_stim.T[np.newaxis, :, :].astype(np.float64, copy=False)

        if self._calibration_params['bad_channels']:
            interpolation_info = self._calibration_params['channel_interpolation_info']
            data[..., interpolation_info['bads_idx'], :] = np.matmul(
                interpolation_info['interpolation_matrix'],
                data[..., interpolation_info['goods_idx'], :],
            )

        filter_opts = self._opts['filter_opts']
        data = _apply_filter(
            data,
            self._calibration_params['qc_filter'],
            filter_opts['pad_time'],
            self._processed_sfreq,
        )

        # Mean subtraction (average reference).
        data -= np.mean(data, axis=1, keepdims=True)

        trial_reject_opts = self._opts['trial_reject_opts']

        trial = data[0]
        qc_stats = self._calibration_params['good_trial_stats_qc']
        pre_reject = trial_reject_opts['pre']
        workspace = self._mad_workspace

        if global_mad_zscore_rejected(
            trial,
            qc_stats['mads_mean'],
            qc_stats['mads_std'],
            pre_reject['global_zscore_threshold'],
            workspace,
        ):
            return None

        if local_mad_zscore_rejected(
            trial,
            pre_reject['local_zscore_threshold'],
            workspace,
        ):
            return None

        model_buffer, _ = crop_eeg_buffer(
            data[0].T,
            np.asarray([self._qc_tmin], dtype=np.float64),
            self._model_tmin,
            self._model_tmax,
            sfreq=self._processed_sfreq,
        )
        return np.ascontiguousarray(model_buffer.T, dtype=np.float64)

    def preprocess_post(
        self,
        eeg_buffer: np.ndarray,
        relative_timestamps: np.ndarray,
    ) -> np.ndarray | None:
        """Resample and preprocess a single post-stimulus trial.

        Must be called after ``calibrate()``.

        Parameters
        ----------
        eeg_buffer : np.ndarray, shape (n_samples, n_channels)
            EEG covering ``[trial_tmin, trial_tmax]`` at ``raw_sfreq``.
        relative_timestamps : np.ndarray, shape (n_samples,)
            Time of each sample in seconds relative to the TMS event.

        Returns a NumPy array with shape (n_channels, n_times), or ``None`` if
        the trial was rejected.
        """
        if self._calibration_params is None:
            raise RuntimeError("calibrate() must be called before preprocessing trials.")

        post_buffer, _ = crop_eeg_buffer(
            eeg_buffer,
            relative_timestamps,
            self._post_initial_tmin,
            self._post_initial_tmax,
        )
        post_buffer = resample_buffer_polyphase(
            post_buffer,
            sfreq_from=self._info["sfreq"],
            sfreq_to=self._processed_sfreq,
        )
        epoch_post = _numpy_to_epochs_array(post_buffer, self._info_processed, self._post_initial_tmin)
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
        for component_idx in calibration_params["ocular_threshold_post"]:
            component_info = calibration_params["ocular_threshold_post"][component_idx]
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
        epoch_post.crop(self._post_tmin, self._post_tmax, include_tmax=True)

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
            tmin=self._post_tmin,
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

        cropped = epoch_post.crop(self._dipole_tmin, self._dipole_tmax, include_tmax=True).get_data(copy=False)
        return np.ascontiguousarray(cropped[0], dtype=np.float64)

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
        if self.qc_epochs is None:
            return 0
        return len(self.qc_epochs)
