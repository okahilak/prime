from functools import lru_cache

import numpy as np
from scipy.signal import firwin, upfirdn


@lru_cache(maxsize=1)
def _cached_lowpass_taps() -> np.ndarray:
    """
    FIR taps of the post-stimulus low-pass: 31 taps, 500 Hz cutoff at 5 kHz, window ('kaiser', 5.0).

    Note: this is the resampler's 5 kHz -> 1 kHz filter (resampling.py, up=1, down=5), which
    band-limited the post window while it ran at 1 kHz, shortened from 101 to 31 taps, so the
    edge of the artifact blank is smeared over +-2.8 ms instead of +-10 ms.
    """
    f_c = 1.0 / 5  # cutoff (relative to Nyquist): 500 Hz at 5 kHz
    return firwin(31, f_c, window=("kaiser", 5.0))


def lowpass_post(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Zero-phase 500 Hz low-pass of 5 kHz post-stimulus epochs along the last axis (time)."""
    if abs(sfreq - 5000.0) > 1e-9:
        raise ValueError(f"the post low-pass is designed for 5000 Hz, got {sfreq}")

    taps = _cached_lowpass_taps()
    half_len = (taps.size - 1) // 2
    # One centred pass with resample_poly's padding ('reflect'): no delay and +-2.8 ms reach
    filtered = upfirdn(taps, data, axis=-1, mode='reflect')
    return filtered[..., half_len:half_len + data.shape[-1]]
