# %%
import numpy as np
import pandas as pd
from scipy.stats import ecdf


class TEPNormalizer:
    """
    A stateful normalizer that learns from a calibration set and applies transformations
    causally to generate soft, probabilistic labels.

    Usage:
        normalizer = TEPNormalizer()
        cal_labels = normalizer.calibrate(cal_amplitudes)
        label = normalizer.transform(new_amplitude)
    """
    def __init__(
        self,
        scale_factor: float = 1.0,
        ewma_span: int = 25,
    ):
        self.scale_factor = scale_factor
        self.ewma_span = ewma_span
        self.warmup_period = ewma_span
        self.is_calibrated = False
        self._cal_mean = 0.0
        self._cal_std = 1.0
        self._cdf_function = None
        self._amplitudes: list = []

    def calibrate(self, cal_amplitudes: np.ndarray) -> np.ndarray:
        """
        Learn normalization stats and ECDF from the calibration block.

        Parameters
        ----------
        cal_amplitudes : array-like
            1-D array of calibration TEP amplitudes (one per trial).

        Returns
        -------
        np.ndarray
            Calibration labels (soft probabilistic values).
        """
        self._amplitudes = list(np.asarray(cal_amplitudes, dtype=float))
        values = pd.Series(self._amplitudes) * self.scale_factor

        ewma_trend = values.ewm(span=self.ewma_span, adjust=True).mean()
        detrended = values - ewma_trend

        stable_detrended = detrended[self.warmup_period:]
        self._cal_mean = np.nanmean(stable_detrended)
        self._cal_std = np.nanstd(stable_detrended)
        if self._cal_std < 1e-9:
            self._cal_std = 1.0

        normalized = (stable_detrended - self._cal_mean) / self._cal_std
        self._cdf_function = ecdf(normalized.dropna()).cdf.evaluate
        self.is_calibrated = True

        all_normalized = (detrended - self._cal_mean) / self._cal_std
        return all_normalized.apply(self._cdf_function).values

    def transform(self, amplitude: float) -> float:
        """
        Transform a single amplitude into a soft label.

        Appends the amplitude to the internal history and recomputes
        the EWMA-based label using the full sequence.

        Parameters
        ----------
        amplitude : float
            A single TEP amplitude value.

        Returns
        -------
        float
            The soft probabilistic label (CDF value).
        """
        if not self.is_calibrated:
            raise RuntimeError("Not calibrated yet. Call .calibrate() first.")

        self._amplitudes.append(float(amplitude))
        values = pd.Series(self._amplitudes) * self.scale_factor

        ewma_trend = values.ewm(span=self.ewma_span, adjust=True).mean()
        detrended = values.iloc[-1] - ewma_trend.iloc[-1]
        normalized = (detrended - self._cal_mean) / self._cal_std
        return float(self._cdf_function(normalized))
