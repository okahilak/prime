"""Butterworth filtering for online preprocessing."""

import numpy as np
from scipy.signal import butter, filtfilt


def butter_filter(data, cutoff, btype, fs, order, pad_time):
    nyquist = fs / 2
    if isinstance(cutoff, list):
        if len(cutoff) == 2:
            normalized = [cutoff[0] / nyquist, cutoff[1] / nyquist]
        else:
            raise ValueError("cutoff should be a single value or a list of length 2.")
    else:
        normalized = cutoff / nyquist
    b, a = butter(order, normalized, btype=btype, analog=False)
    filtered = apply_filter(data, [b, a], pad_time, fs)
    return filtered, [b, a]


def apply_filter(data, coeffs, pad_time, fs):
    n_pad = int(pad_time * fs)
    padded = np.pad(data, ((0, 0), (0, 0), (n_pad, n_pad)), mode='reflect')
    filtered = filtfilt(coeffs[0], coeffs[1], padded, padlen=None)
    return filtered[:, :, n_pad:-n_pad]


def apply_filter_2d(
    data: np.ndarray,
    coeffs,
    n_pad: int,
    workspace: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``apply_filter`` for a single trial shaped (n_channels, n_times).

    Returns ``(filtered, workspace)`` so callers can reuse the reflect-pad buffer.
    """
    n_ch, n_times = data.shape
    padded_len = n_times + 2 * n_pad
    if workspace is None or workspace.shape != (n_ch, padded_len):
        workspace = np.empty((n_ch, padded_len), dtype=np.float64)
    workspace[:, n_pad:n_pad + n_times] = data
    if n_pad:
        workspace[:, :n_pad] = data[:, 1:n_pad + 1][:, ::-1]
        workspace[:, n_pad + n_times:] = data[:, -(n_pad + 1):-1][:, ::-1]
    filtered = filtfilt(coeffs[0], coeffs[1], workspace, axis=-1, padlen=None)
    return filtered[:, n_pad:-n_pad], workspace
