"""Fast median-absolute-deviation checks for single-trial online rejection."""

from __future__ import annotations

import numpy as np


def _median_1d(x: np.ndarray) -> float:
    """Median of a 1-D array via ``np.partition`` (faster than ``np.median``)."""
    n = x.size
    if n == 0:
        return np.nan
    if n % 2:
        return float(np.partition(x, n // 2)[n // 2])
    idx = (n // 2 - 1, n // 2)
    part = np.partition(x, idx)
    return 0.5 * (float(part[idx[0]]) + float(part[idx[1]]))


def median_abs_deviation_global(trial: np.ndarray, workspace: SingleTrialMadWorkspace | None = None) -> float:
    """Global MAD over channels and time for one trial.

    Matches ``scipy.stats.median_abs_deviation(trial[np.newaxis], axis=(1, 2))[0]``.
    """
    if workspace is None:
        flat = trial.ravel()
        center = _median_1d(flat)
        return _median_1d(np.abs(flat - center))

    np.copyto(workspace._flat, trial.ravel())
    center = _median_1d(workspace._flat)
    np.subtract(workspace._flat, center, out=workspace._dev)
    np.abs(workspace._dev, out=workspace._dev)
    return _median_1d(workspace._dev)


def median_abs_deviation_per_channel(
    trial: np.ndarray,
    workspace: SingleTrialMadWorkspace | None = None,
) -> np.ndarray:
    """Per-channel MAD along time for one trial, shape ``(n_channels,)``.

    Matches ``scipy.stats.median_abs_deviation(trial[np.newaxis], axis=2)[0]``.
    """
    n_channels = trial.shape[0]
    if workspace is None:
        out = np.empty(n_channels, dtype=trial.dtype)
    else:
        out = workspace._per_channel

    for channel in range(n_channels):
        row = trial[channel]
        center = _median_1d(row)
        out[channel] = _median_1d(np.abs(row - center))
    return out


def global_mad_zscore_rejected(
    trial: np.ndarray,
    mads_mean: float,
    mads_std: float,
    threshold: tuple[float, float],
    workspace: SingleTrialMadWorkspace | None = None,
) -> bool:
    """Return True if the trial fails the global MAD z-score gate."""
    mad_val = median_abs_deviation_global(trial, workspace)
    z_mad = (mad_val - mads_mean) / mads_std
    return z_mad < threshold[0] or z_mad > threshold[1]


def local_mad_zscore_rejected(
    trial: np.ndarray,
    local_zscore_threshold: float,
    workspace: SingleTrialMadWorkspace | None = None,
) -> bool:
    """Return True if any channel fails the local MAD z-score gate."""
    local_mad = median_abs_deviation_per_channel(trial, workspace)
    std = local_mad.std()
    if std == 0.0:
        return False
    mean = local_mad.mean()
    return bool(np.any(np.abs(local_mad - mean) > local_zscore_threshold * std))


class SingleTrialMadWorkspace:
    """Reusable buffers for repeated single-trial MAD checks."""

    __slots__ = ("_flat", "_dev", "_per_channel")

    def __init__(self, n_channels: int, n_times: int):
        n_samples = n_channels * n_times
        self._flat = np.empty(n_samples, dtype=np.float64)
        self._dev = np.empty(n_samples, dtype=np.float64)
        self._per_channel = np.empty(n_channels, dtype=np.float64)
