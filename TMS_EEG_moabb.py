# %%
import logging
import re
from pathlib import Path
from typing import List, Optional, Union

import mne
import numpy as np
import pandas as pd
from mne import BaseEpochs
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from tqdm.auto import tqdm
from scipy.stats import ecdf
import matplotlib.pyplot as plt

# Setup logger for the entire module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

DATA_ROOT_PATH = Path("~/prime-data/processed").expanduser()

# %%
# Real-Time Compatible TEP Normalizer with Warm-up
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


# %%
class TMSEEGDataset(BaseDataset):
    """Base dataset for preprocessed TMS-EEG data."""
    def __init__(self, data_path: Union[str, Path, None] = None, subject_list: Union[List[int], None] = None):
        self.data_path_root = Path(data_path) if data_path else DATA_ROOT_PATH
        effective_subject_list = subject_list if subject_list is not None else self._discover_subjects()
        super().__init__(
            subjects=effective_subject_list, sessions_per_subject=1, events={"TMS_stim": 1},
            code="TMSEEGDataset", interval=[-0.505, -0.006], paradigm="generic_tms_eeg", doi=None
        )

    def _discover_subjects(self) -> List[int]:
        subjects = set()
        if not self.data_path_root.is_dir(): return []
        for subject_dir in self.data_path_root.glob("sub-*"):
            if subject_dir.is_dir():
                match = re.search(r"sub-(\d+)", subject_dir.name)
                if match: subjects.add(int(match.group(1)))
        return sorted(list(subjects))

    def _get_single_subject_data(self, subject: int) -> dict:
        raise NotImplementedError("Subclasses must implement _get_single_subject_data")

    def data_path(self, subject: int, **kwargs) -> List[str]:
        subject_id_str = f"{subject:03d}"
        paths = [
            self.data_path_root / f"sub-{subject_id_str}" / f"sub-{subject_id_str}_calibration_pre.fif",
            self.data_path_root / f"sub-{subject_id_str}" / f"sub-{subject_id_str}_intervention_pre.fif",
        ]
        return [str(p) for p in paths if p.exists()]


class TMSEEGDatasetTEPfree(TMSEEGDataset):
    """Dataset for free-orientation TEP data, reusing the TMSEEGDataset structure."""
    def __init__(self, data_path: Union[str, Path, None] = None, subject_list: Union[List[int], None] = None):
        super().__init__(data_path, subject_list)
        self.code = "TMSEEGDatasetTEPfree"

    def _get_single_subject_data(self, subject: int) -> dict:
        subject_id_str = f"{subject:03d}"
        subj_dir = self.data_path_root / f"sub-{subject_id_str}"

        # Load calibration and intervention files separately
        eeg_cal_file = subj_dir / f"sub-{subject_id_str}_calibration_pre.fif"
        eeg_int_file = subj_dir / f"sub-{subject_id_str}_intervention_pre.fif"
        tep_cal_file = subj_dir / f"sub-{subject_id_str}_calibration_dipoles.npz"
        tep_int_file = subj_dir / f"sub-{subject_id_str}_intervention_dipoles.npz"

        required_files = [eeg_cal_file, eeg_int_file, tep_cal_file, tep_int_file]
        if not all(f.exists() for f in required_files):
            log.warning(f"Data files missing for S{subject} (TEP). Skipping.")
            return {}

        epochs_cal = mne.read_epochs(eeg_cal_file, preload=True, verbose=False)
        epochs_int = mne.read_epochs(eeg_int_file, preload=True, verbose=False)

        try:
            npz_cal = np.load(tep_cal_file, allow_pickle=True)
            dipoles_cal = npz_cal['trial_dipoles_free_ori']
            tep_cal = np.array([d['amplitude'] for d in dipoles_cal]).flatten()

            npz_int = np.load(tep_int_file, allow_pickle=True)
            dipoles_int = npz_int['trial_dipoles_free_ori']
            tep_int = np.array([d['amplitude'] for d in dipoles_int]).flatten()
        except Exception as e:
            log.error(f"S{subject}: Error loading TEP file: {e}", exc_info=True)
            return {}

        n_cal = len(epochs_cal)
        n_int = len(epochs_int)

        if n_cal != len(tep_cal) or n_int != len(tep_int):
            log.error(f"S{subject}: Mismatch in TEP data lengths. Skipping.")
            return {}

        # Concatenate calibration + intervention
        epochs = mne.concatenate_epochs([epochs_cal, epochs_int])
        tep_amplitudes = np.concatenate([tep_cal, tep_int])
        period_labels = np.array(['calibration'] * n_cal + ['intervention'] * n_int)

        epochs.metadata = pd.DataFrame({
            "TEP_amplitude": tep_amplitudes,
            "period": period_labels,
        })
        return {"0": {"0": epochs}}


# %%
class _BaseTMSEEGParadigm(BaseParadigm):
    """
    Base class for TMS-EEG paradigms, uses a fixed number of initial
    trials for calibration.
    """
    def __init__(
        self,
        tmin: float,
        tmax: float,
        target_metadata_col: str,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(filters=[], **kwargs)
        self.tmin = tmin
        self.tmax = tmax
        self.target_metadata_col = target_metadata_col
        self.calibration_trials = 100

    @property
    def datasets(self):
        return [TMSEEGDataset()]

    def is_valid(self, dataset):
        return "tms_eeg" in dataset.paradigm

    def used_events(self, dataset):
        return dict(dataset.event_id)

    def make_normalizer_pipeline(self):
        raise NotImplementedError("Subclasses must implement their own normalizer pipeline.")

    def get_data(self, dataset, subjects=None, return_epochs=False):
        """
        Main method to retrieve and process data.
        """
        if not self.is_valid(dataset):
            raise ValueError(f"Dataset {dataset.code} is not compatible.")

        subject_list = subjects if subjects is not None else dataset.subject_list
        raw_epochs_data = dataset.get_data(subject_list)

        X_list, y_list, metadata_list = [], [], []

        for subject in tqdm(subject_list, desc=f"Processing subjects for {self.__class__.__name__}"):
            if subject not in raw_epochs_data:
                continue

            epochs = raw_epochs_data[subject]["0"]["0"]
            epochs.crop(tmin=self.tmin, tmax=self.tmax, include_tmax=True)
            
            full_metadata = epochs.metadata.copy()

            # Determine calibration trials from metadata 'period' column
            cal_mask = full_metadata['period'] == 'calibration'
            n_cal = cal_mask.sum()
            if n_cal == 0:
                log.warning(f"S{subject}: No calibration trials found in metadata. Skipping.")
                continue
            meta_calibration = full_metadata[cal_mask]

            normalizer = self.make_normalizer_pipeline()
            normalizer.fit(meta_calibration)
            y_run = normalizer.transform(full_metadata)

            nan_mask = np.isnan(y_run)
            if np.any(nan_mask):
                log.warning(f"S{subject}: Found {np.sum(nan_mask)} NaN labels. Removing corresponding trials.")
                epochs = epochs[~nan_mask]
                y_run = y_run[~nan_mask]
                full_metadata = full_metadata[~nan_mask]

            if len(epochs) == 0:
                continue

            y_list.append(y_run)
            metadata_list.append(full_metadata)
            X_list.append(epochs.get_data(copy=False) if not return_epochs else epochs)

        if not X_list:
            return np.array([]), np.array([]), pd.DataFrame()

        metadata_final = pd.concat(metadata_list, ignore_index=True)
        y_final = np.concatenate(y_list)
        X_final = np.concatenate(X_list, axis=0) if not return_epochs else mne.concatenate_epochs(X_list)

        log.info(f"Final data shapes - X: {X_final.shape}, y: {y_final.shape}")
        return X_final, y_final, metadata_final


class TMSEEGClassificationTEPfree(_BaseTMSEEGParadigm):
    """Classification paradigm for free-orientation TEP data."""

    def __init__(self, tmin: float = -0.5, tmax: float = -0.020, **kwargs):
        super().__init__(
            tmin=tmin,
            tmax=tmax,
            target_metadata_col="TEP_amplitude",
            **kwargs,
        )

    @property
    def scoring(self):
        return "roc_auc"

    @property
    def datasets(self):
        return [TMSEEGDatasetTEPfree()]

    def make_normalizer_pipeline(self):
        return TEPNormalizer(target_col=self.target_metadata_col, scale_factor=1.0)


# %%
def plot_tep_normalizer_diagnostics(subject_id, metadata, final_labels, target_col, scale_factor=1.0, ewma_span=25):
    """Generates a multi-panel plot to visualize the new normalization process."""
    warmup_period = ewma_span 

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"Real-Time Normalization Diagnostics for Subject {subject_id} ({target_col})", fontsize=16)

    raw_values = metadata[target_col].values * scale_factor
    blocks = metadata['block'].values
    trials = np.arange(len(raw_values))
    
    ewma_trend = pd.Series(raw_values).ewm(span=ewma_span, adjust=True).mean()
    detrended_values = raw_values - ewma_trend

    # Panel 1: Raw Data and Adaptive Trend
    scatter1 = axes[0].scatter(trials, raw_values, c=blocks, cmap='viridis', alpha=0.5, s=15, label="Raw Values")
    axes[0].plot(trials[warmup_period:], ewma_trend[warmup_period:], color='red', lw=2, label=f'Stable EWMA Trend')
    axes[0].set_title("1. Raw Values and Causally-Calculated Adaptive Trend")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle="--")
    axes[0].legend()
    fig.colorbar(scatter1, ax=axes[0], label='Block ID')

    # Panel 2: Detrended Data
    scatter2 = axes[1].scatter(trials, detrended_values, c=blocks, cmap='viridis', alpha=0.7, s=15)
    axes[1].axhline(0, color='red', lw=2, linestyle='--', label='Zero Line')
    axes[1].set_title("2. Detrended Values (Raw - EWMA Trend)")
    axes[1].set_ylabel("Detrended Amplitude")
    axes[1].grid(True, linestyle="--")
    axes[1].legend()

    # Panel 3: Final Soft Labels
    scatter3 = axes[2].scatter(trials, final_labels, c=blocks, cmap='viridis', alpha=0.7, s=15)
    axes[2].set_title("3. Final Probabilistic 'Soft' Labels (from ECDF of stable part of Block 1)")
    axes[2].set_xlabel("Trial Number")
    axes[2].set_ylabel("Soft Label [0-1]")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, linestyle="--")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # --- TEP ANALYSIS (Example) ---
    subject_tep = 106
    log.info(f"\n{'='*25} RUNNING TEP EXAMPLE (Subject {subject_tep}) {'='*25}")
    dataset_tep = TMSEEGDatasetTEPfree(subject_list=[subject_tep])

    if not dataset_tep.subject_list:
        log.error(f"Subject {subject_tep} not found for TEP task.")
    else:
        paradigm_tep = TMSEEGClassificationTEPfree(tmin=-0.5, tmax=-0.020)
        X_tep, y_tep, meta_tep = paradigm_tep.get_data(dataset_tep)

        if y_tep.size > 0:
            log.info("Generating diagnostic plot for TEP processing...")
            plot_tep_normalizer_diagnostics(
                subject_id=subject_tep,
                metadata=meta_tep,
                final_labels=y_tep,
                target_col='TEP_amplitude',
                scale_factor=1.0
            )
        else:
            log.warning("No TEP data available to plot.")


# %%
