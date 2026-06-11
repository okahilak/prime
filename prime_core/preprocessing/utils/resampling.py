import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
from scipy.signal import firwin, resample_poly


@lru_cache(maxsize=8)
def _cached_resample_up_down(sfreq_from_str: str, sfreq_to_str: str) -> tuple[int, int]:
    ratio = Fraction(sfreq_to_str) / Fraction(sfreq_from_str)
    up = ratio.numerator
    down = ratio.denominator
    g = math.gcd(up, down)
    # resample_poly reduces by gcd as well, so we mirror that here.
    return up // g, down // g


@lru_cache(maxsize=16)
def _cached_resample_window_taps(
    up: int, down: int, dtype_name: str
) -> np.ndarray:
    """
    Unscaled FIR taps for resample_poly's default window ('kaiser', 5.0).

    Note: resample_poly applies `h *= up` internally after designing/accepting the taps.
    """
    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff (relative to Nyquist)
    half_len = 10 * max_rate  # filter half-length
    taps_len = 2 * half_len + 1
    dtype = np.dtype(dtype_name)
    return firwin(taps_len, f_c, window=("kaiser", 5.0)).astype(dtype)


def resample_buffer_polyphase(
    data: np.ndarray,
    sfreq_from: float,
    sfreq_to: float,
) -> np.ndarray:
    """Resample (n_samples, n_channels) using scipy polyphase."""
    if abs(sfreq_from - sfreq_to) < 1e-9:
        return data

    up, down = _cached_resample_up_down(str(sfreq_from), str(sfreq_to))
    if up == down == 1:
        return data.copy()

    window_taps = _cached_resample_window_taps(up, down, data.dtype.name)
    return resample_poly(
        data,
        up=up,
        down=down,
        axis=0,
        window=window_taps,
        padtype='reflect',
    )
