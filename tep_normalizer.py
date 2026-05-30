# %%
import numpy as np
from scipy.stats import ecdf


class TEPNormalizer:
    """
    A stateful normalizer that learns from a calibration set and applies transformations
    causally to generate soft, probabilistic labels.

    Call transform() one amplitude at a time; EWMA state is maintained internally.
    """
    def __init__(
        self,
        scale_factor: float = 1.0,
        ewma_span: int = 25,
    ):
        self.scale_factor = scale_factor
        self.ewma_span = ewma_span
        self._alpha = 2.0 / (ewma_span + 1)
        # --- Ignore the initial unstable period of EWMA when fitting ---
        self.warmup_period = ewma_span
        self.is_fitted = False
        self.cal_mean_ = 0
        self.cal_std_ = 1
        self.cdf_function_ = None
        # EWMA running state (for adjust=True mode)
        self._ewma_numer = 0.0
        self._ewma_denom = 0.0

    def fit(self, cal_amplitudes: np.ndarray):
        """
        Learns normalization stats and the ECDF from the calibration block,
        ignoring the initial EWMA warm-up period for stability.

        Also resets the EWMA running state so subsequent transform() calls
        start fresh.

        Parameters
        ----------
        cal_amplitudes : array-like
            1-D array of calibration TEP amplitudes (one per trial).
        """
        import pandas as pd
        values = pd.Series(np.asarray(cal_amplitudes, dtype=float)) * self.scale_factor

        ewma_trend = values.ewm(span=self.ewma_span, adjust=True).mean()
        detrended_values = values - ewma_trend

        # --- Use warmup period to learn from the stable part of the signal ---
        stable_detrended_values = detrended_values[self.warmup_period:]

        self.cal_mean_ = np.nanmean(stable_detrended_values)
        self.cal_std_ = np.nanstd(stable_detrended_values)
        if self.cal_std_ < 1e-9:
            self.cal_std_ = 1

        normalized_values = (stable_detrended_values - self.cal_mean_) / self.cal_std_
        self.cdf_function_ = ecdf(normalized_values.dropna()).cdf.evaluate
        self.is_fitted = True
        self._ewma_numer = 0.0
        self._ewma_denom = 0.0
        return self

    def transform(self, amplitude: float) -> float:
        """
        Transform a single amplitude into a soft label, updating EWMA state.

        Parameters
        ----------
        amplitude : float
            A single TEP amplitude value.

        Returns
        -------
        float
            The soft probabilistic label (CDF value).
        """
        if not self.is_fitted:
            raise RuntimeError("The normalizer has not been fitted yet. Call .fit() first.")

        value = float(amplitude) * self.scale_factor

        # Update EWMA (pandas adjust=True equivalent):
        # numer_t = x_t + (1 - alpha) * numer_{t-1}
        # denom_t = 1 + (1 - alpha) * denom_{t-1}
        # ewma_t = numer_t / denom_t
        decay = 1.0 - self._alpha
        self._ewma_numer = value + decay * self._ewma_numer
        self._ewma_denom = 1.0 + decay * self._ewma_denom
        ewma = self._ewma_numer / self._ewma_denom

        detrended = value - ewma
        normalized = (detrended - self.cal_mean_) / self.cal_std_
        return float(self.cdf_function_(normalized))
