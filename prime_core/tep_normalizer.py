# %%
import numpy as np
import pandas as pd
from scipy.stats import ecdf


class TEPNormalizer:
    """
    A stateful normalizer that learns from a calibration set and applies transformations
    causally to generate soft, probabilistic labels.

    Label = ECDF of the EWMA-detrended amplitude 
    EWMA continuously updated 
    ECDF frozen after calibration

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
        self._cdf_function = ecdf(stable_detrended.dropna()).cdf.evaluate
        self.is_calibrated = True

        return detrended.apply(self._cdf_function).values

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
        return float(self._cdf_function(detrended))

    def transform_verbose(self, amplitude: float) -> dict:
        """Same arithmetic and same state mutation as transform(),
        but returns every intermediate instead of only the final label."""
        if not self.is_calibrated:
            raise RuntimeError("Not calibrated yet. Call .calibrate() first.")

        self._amplitudes.append(float(amplitude))
        values = pd.Series(self._amplitudes) * self.scale_factor

        ewma_trend = values.ewm(span=self.ewma_span, adjust=True).mean()
        trend      = float(ewma_trend.iloc[-1])
        detrended  = float(values.iloc[-1] - trend)

        return {
            "raw":       float(amplitude),
            "ewma":      trend,
            "detrended": detrended,
            "label":     float(self._cdf_function(detrended)),
        }
