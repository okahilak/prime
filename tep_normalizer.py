# %%
import numpy as np
import pandas as pd
from scipy.stats import ecdf


class TEPNormalizer:
    """
    A stateful normalizer that learns from a calibration set and applies transformations
    causally to generate soft, probabilistic labels.
    """
    def __init__(
        self,
        target_col: str,
        scale_factor: float = 1.0,
        ewma_span: int = 25,
    ):
        self.target_col = target_col
        self.scale_factor = scale_factor
        self.ewma_span = ewma_span
        # --- Ignore the initial unstable period of EWMA when fitting ---
        self.warmup_period = ewma_span
        self.is_fitted = False
        self.cal_mean_ = 0
        self.cal_std_ = 1
        self.cdf_function_ = None

    def fit(self, metadata_df_cal: pd.DataFrame):
        """
        Learns normalization stats and the ECDF from the calibration block,
        ignoring the initial EWMA warm-up period for stability.
        """
        metadata_copy = metadata_df_cal.copy()
        values = metadata_copy[self.target_col].astype(float) * self.scale_factor

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
        return self

    def transform(self, metadata_df_full: pd.DataFrame) -> np.ndarray:
        """
        Applies the learned transformations to the full session's data.
        """
        if not self.is_fitted:
            raise RuntimeError("The normalizer has not been fitted yet. Call .fit() first.")

        metadata_copy = metadata_df_full.copy()
        values = metadata_copy[self.target_col].astype(float) * self.scale_factor
        
        ewma_trend = values.ewm(span=self.ewma_span, adjust=True).mean()
        detrended_values = values - ewma_trend
        
        normalized_values = (detrended_values - self.cal_mean_) / self.cal_std_
        soft_labels = normalized_values.apply(self.cdf_function_)
        
        return soft_labels.values
