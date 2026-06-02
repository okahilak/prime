# %%
"""
Datasets Module for TMS-EEG Classification.

Core Components:
- `TEPDataset`: MOABB-compatible dataset for free-orientation TEP data.
- `TEPParadigm`: Paradigm that loads TEP data and generates soft labels.
- `EEGDataset`: A PyTorch-compatible Dataset class.
- `load_pretrain_data`: Main function to load, preprocess, align, and
  concatenate data from multiple subjects.
- `TEPParadigmWithAblation` handles the specifics of data
  extraction for TMS-EEG data types.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from moabb.datasets.base import BaseDataset
from moabb.paradigms.base import BaseParadigm
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from prime.tep_normalizer import TEPNormalizer
from prime.tta_wrapper import (
    PYRIEMANN_AVAILABLE,
    _apply_alignment_transform_np,
    _compute_alignment_transform_np,
    _compute_reference_covariance_np,
    _compute_trial_covariances_np,
)

# --- Module-level Logger Setup ---
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"


# %%
# ----------------------------------------------------------------------------
# MOABB DATASET
# ----------------------------------------------------------------------------

class TEPDataset(BaseDataset):
    """Dataset for free-orientation TEP data."""
    def __init__(self, data_path: Union[str, Path, None] = None, subject_list: Union[List[int], None] = None):
        self.data_path_root = Path(data_path) if data_path else DATA_ROOT_PATH
        effective_subject_list = subject_list if subject_list is not None else self._discover_subjects()
        super().__init__(
            subjects=effective_subject_list, sessions_per_subject=1, events={"TMS_stim": 1},
            code="TEPDataset", interval=[-0.505, -0.006], paradigm="generic_tms_eeg", doi=None
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
        subject_id_str = f"{subject:03d}"
        subj_dir = self.data_path_root / f"sub-{subject_id_str}"

        eeg_cal_file = subj_dir / f"sub-{subject_id_str}_calibration_pre.fif"
        eeg_int_file = subj_dir / f"sub-{subject_id_str}_intervention_pre.fif"
        tep_cal_file = subj_dir / f"sub-{subject_id_str}_calibration_amplitudes.npy"
        tep_int_file = subj_dir / f"sub-{subject_id_str}_intervention_amplitudes.npy"

        if not eeg_cal_file.exists():
            raise FileNotFoundError(f"S{subject}: Missing calibration EEG file: {eeg_cal_file}")
        if not eeg_int_file.exists():
            raise FileNotFoundError(f"S{subject}: Missing intervention EEG file: {eeg_int_file}")
        if not tep_cal_file.exists():
            raise FileNotFoundError(f"S{subject}: Missing calibration TEP file: {tep_cal_file}")
        if not tep_int_file.exists():
            raise FileNotFoundError(f"S{subject}: Missing intervention TEP file: {tep_int_file}")

        epochs_cal = mne.read_epochs(eeg_cal_file, preload=True, verbose=False)
        epochs_int = mne.read_epochs(eeg_int_file, preload=True, verbose=False)

        try:
            tep_cal = np.load(tep_cal_file)
            tep_int = np.load(tep_int_file)
        except Exception as e:
            log.error(f"S{subject}: Error loading TEP file: {e}", exc_info=True)
            return {}

        n_cal = len(epochs_cal)
        n_int = len(epochs_int)

        if n_cal != len(tep_cal) or n_int != len(tep_int):
            log.error(f"S{subject}: Mismatch in TEP data lengths. Skipping.")
            return {}

        epochs = mne.concatenate_epochs([epochs_cal, epochs_int])
        tep_amplitudes = np.concatenate([tep_cal, tep_int])
        period_labels = np.array(['calibration'] * n_cal + ['intervention'] * n_int)

        epochs.metadata = pd.DataFrame({
            "TEP_amplitude": tep_amplitudes,
            "period": period_labels,
        })
        return {"0": {"0": epochs}}

    def data_path(self, subject: int, **kwargs) -> List[str]:
        subject_id_str = f"{subject:03d}"
        paths = [
            self.data_path_root / f"sub-{subject_id_str}" / f"sub-{subject_id_str}_calibration_pre.fif",
            self.data_path_root / f"sub-{subject_id_str}" / f"sub-{subject_id_str}_intervention_pre.fif",
        ]
        return [str(p) for p in paths if p.exists()]


# %%
# ----------------------------------------------------------------------------
# MOABB PARADIGM
# ----------------------------------------------------------------------------

class TEPParadigm(BaseParadigm):
    """Paradigm for free-orientation TEP classification with soft labels."""

    def __init__(
        self,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(filters=[], **kwargs)
        self.fmin = fmin
        self.fmax = fmax
        self.target_metadata_col = "TEP_amplitude"

    @property
    def scoring(self):
        return "roc_auc"

    @property
    def datasets(self):
        return [TEPDataset()]

    def is_valid(self, dataset):
        return "tms_eeg" in dataset.paradigm

    def used_events(self, dataset):
        return dict(dataset.event_id)

    def _extract_amplitudes(self, full_metadata, cal_mask):
        """Extract calibration and full amplitude arrays from metadata."""
        cal_amplitudes = full_metadata.loc[cal_mask, self.target_metadata_col].values.astype(float)
        all_amplitudes = full_metadata[self.target_metadata_col].values.astype(float)
        return cal_amplitudes, all_amplitudes

    def get_data(self, dataset, subjects=None, return_epochs=False):
        if not self.is_valid(dataset):
            raise ValueError(f"Dataset {dataset.code} is not compatible.")

        subject_list = subjects if subjects is not None else dataset.subject_list
        raw_epochs_data = dataset.get_data(subject_list)

        X_list, y_list, metadata_list = [], [], []

        for subject in tqdm(subject_list, desc="Processing subjects"):
            if subject not in raw_epochs_data:
                continue

            epochs = raw_epochs_data[subject]["0"]["0"]

            full_metadata = epochs.metadata.copy()

            cal_mask = full_metadata['period'] == 'calibration'
            n_cal = cal_mask.sum()
            if n_cal == 0:
                log.warning(f"S{subject}: No calibration trials found in metadata. Skipping.")
                continue

            cal_amplitudes, all_amplitudes = self._extract_amplitudes(
                full_metadata, cal_mask)
            normalizer = TEPNormalizer(scale_factor=1.0)
            cal_labels = normalizer.calibrate(cal_amplitudes)
            int_labels = np.array([normalizer.transform(a) for a in all_amplitudes[len(cal_amplitudes):]])
            y_run = np.concatenate([cal_labels, int_labels])

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


# %%
# ----------------------------------------------------------------------------
# PARADIGM AND DATASET CONFIGURATION
# ----------------------------------------------------------------------------

# TEP dataset specs (sampling rate, duration, number of classes)
TEP_SAMPLE_RATE = 1000
TEP_DURATION_SEC = 0.995
TEP_N_CLASSES = 2

# %%
class EEGDataset(Dataset):
    """A PyTorch-compatible dataset for EEG epochs."""

    def __init__(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        channel_names: Optional[List[str]] = None,
    ):
        """
        Initializes the dataset.
        """
        self.epochs = epochs
        self.labels = labels
        self.channel_names = channel_names

    def __len__(self) -> int:
        """Return the total number of trials in the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Retrieves a single trial and its associated label.
        """
        sample = {
            "epoch": self.epochs[index],
            "label": self.labels[index],
        }
        if self.channel_names is not None:
            sample["channel_names"] = self.channel_names
        return sample


# %%
# DATA LOADING AND MANAGEMENT FUNCTIONS

def get_subject_list(data_root: str) -> List[int]:
    """Retrieves a sorted list of subject IDs for the TEP dataset."""
    os.environ["MNE_DATA"] = data_root
    os.environ["MNE_DATASETS_MOABB_PATH"] = data_root

    dataset_instance = TEPDataset(data_path=data_root)
    subjects = sorted(dataset_instance.subject_list)
    log.info(f"Found {len(subjects)} subjects.")
    return subjects


def load_pretrain_data(
    subject_ids: List[int],
    paradigm_kwargs: Dict[str, Any],
    data_root: str,
    args: Any,
    apply_trial_ablation: bool = False,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    int,
    int,
    Optional[List[str]],
    Optional[np.ndarray],
]:
    """Loads, processes, and optionally aligns TEP EEG data for pre-training."""
    os.environ["MNE_DATA"] = data_root
    os.environ["MNE_DATASETS_MOABB_PATH"] = data_root

    effective_paradigm_kwargs = paradigm_kwargs.copy()
    if apply_trial_ablation and hasattr(args, "num_trials_per_subject"):
        num_trials = args.num_trials_per_subject
        effective_paradigm_kwargs["num_trials_per_subject"] = num_trials
        log.info(f"Applying trial ablation: {num_trials} trials per subject.")

    paradigm = TEPParadigmWithAblation(**effective_paradigm_kwargs)
    dataset = TEPDataset(data_path=data_root)

    available_subjects = dataset.subject_list
    subjects_to_load = sorted(s for s in subject_ids if s in available_subjects)
    if not subjects_to_load:
        log.warning("No requested subjects found in dataset.")
        return None, None, 0, 0, None, None

    log.info(f"Loading data for {len(subjects_to_load)} subjects.")
    epochs_data, y_values, _ = paradigm.get_data(
        dataset=dataset, subjects=subjects_to_load
    )

    if epochs_data is None or epochs_data.size == 0:
        log.warning("No data returned from paradigm.")
        return None, None, 0, 0, None, None

    # --- Label and Data Shape Processing ---
    labels_numeric = np.array(y_values).astype(np.float64)
    log.info(f"Using probabilistic float labels (dtype: {labels_numeric.dtype}).")
    _, n_channels, n_timepoints = epochs_data.shape

    if hasattr(paradigm, "channels") and paradigm.channels:
        channel_names = paradigm.channels
    else:
        channel_names = [f"Ch{i+1}" for i in range(n_channels)]

    all_epochs_list = [epochs_data]
    all_labels_list = [labels_numeric]

    # --- Euclidean Alignment (EA) Processing ---
    use_alignment = (
        hasattr(args, "use_tta")
        and args.use_tta
        and hasattr(args, "alignment_type")
        and args.alignment_type != "none"
    )

    if not use_alignment:
        log.info("Skipping Euclidean Alignment.")
        concatenated_epochs = np.concatenate(all_epochs_list)
        concatenated_labels = np.concatenate(all_labels_list)
        return (
            concatenated_epochs,
            concatenated_labels,
            n_channels,
            n_timepoints,
            channel_names,
            None,
        )

    log.info(f"Applying per-subject alignment (type: {args.alignment_type})")
    global_backrotation = None
    use_backrotation = getattr(args, "use_backrotation", False)

    # 1. Compute global back-rotation matrix if requested
    if use_backrotation:
        all_trial_covs = [
            _compute_trial_covariances_np(_subj_epochs, args.alignment_cov_epsilon)
            for _subj_epochs in all_epochs_list
            if _subj_epochs.size
        ]
        if all_trial_covs:
            trial_covs_np = np.concatenate(all_trial_covs, axis=0)
            align_type = args.alignment_type
            if align_type == "riemannian" and not PYRIEMANN_AVAILABLE:
                align_type = "euclidean"

            Sigma_global_np = _compute_reference_covariance_np(
                trial_covs_np, alignment_type=align_type
            )
            eps_glob = max(
                args.alignment_cov_epsilon, np.trace(Sigma_global_np) * 1e-6
            )
            Sigma_global_np += eps_glob * np.eye(Sigma_global_np.shape[0])
            eigvals, eigvecs = np.linalg.eigh(Sigma_global_np)
            global_backrotation = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    # 2. Apply alignment to each subject's data block
    aligned_epochs_list = []
    for subject_epochs_np in all_epochs_list:
        if subject_epochs_np.size == 0:
            continue
        source_trial_covs = _compute_trial_covariances_np(
            subject_epochs_np, args.alignment_cov_epsilon
        )
        Sigma_s_np = _compute_reference_covariance_np(
            source_trial_covs, alignment_type=args.alignment_type
        )
        R_s_neg_half = _compute_alignment_transform_np(
            Sigma_s_np, args.alignment_transform_epsilon
        )
        aligned_subject_epochs = _apply_alignment_transform_np(
            subject_epochs_np, R_s_neg_half
        )

        if use_backrotation and global_backrotation is not None:
            aligned_subject_epochs = _apply_alignment_transform_np(
                aligned_subject_epochs, global_backrotation
            )
        aligned_epochs_list.append(aligned_subject_epochs)

    log.info("Pretraining alignment complete.")

    # --- Final Concatenation ---
    concatenated_epochs = (
        np.concatenate(aligned_epochs_list) if aligned_epochs_list else np.array([])
    )
    concatenated_labels = (
        np.concatenate(all_labels_list) if all_labels_list else np.array([])
    )
    log.info(
        f"Final Shapes: Epochs {concatenated_epochs.shape}, Labels {concatenated_labels.shape}"
    )

    return (
        concatenated_epochs,
        concatenated_labels,
        n_channels,
        n_timepoints,
        channel_names,
        global_backrotation,
    )


# %%
# PARADIGM WRAPPER CLASSES

class TEPParadigmWithAblation(TEPParadigm):
    """TEPParadigm with optional per-subject trial ablation."""

    def __init__(self, num_trials_per_subject: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._num_trials_per_subject = num_trials_per_subject

    def get_data(
        self, dataset: BaseDataset, subjects: Optional[List[int]] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load data for all subjects, optionally truncating per-subject trials."""
        if subjects is None:
            subjects = dataset.subject_list

        all_X, all_y, all_meta = [], [], []

        for subject in subjects:
            X, y, metadata = super().get_data(dataset=dataset, subjects=[subject])

            if self._num_trials_per_subject is not None and len(X) > self._num_trials_per_subject:
                log.info(f"Slicing data for Subj {subject} to {self._num_trials_per_subject} trials.")
                X = X[: self._num_trials_per_subject]
                y = y[: self._num_trials_per_subject]
                metadata = metadata.iloc[: self._num_trials_per_subject]

            all_X.append(X)
            all_y.append(y)
            all_meta.append(metadata)

        final_X = np.concatenate(all_X) if all_X else np.array([])
        final_y = np.concatenate(all_y) if all_y else np.array([])
        final_meta = (
            pd.concat(all_meta, ignore_index=True) if all_meta else pd.DataFrame()
        )

        return final_X, final_y, final_meta


# %%
# MODULE EXPORTS

__all__ = [
    "TEPDataset",
    "TEPParadigm",
    "EEGDataset",
    "get_subject_list",
    "load_pretrain_data",
    "TEPParadigmWithAblation",
    "TEP_SAMPLE_RATE",
    "TEP_DURATION_SEC",
    "TEP_N_CLASSES",
]
# %%
