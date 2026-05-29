# %%
"""
Datasets Module for TMS-EEG Classification.

Core Components:
- `EEGDataset`: A PyTorch-compatible Dataset class.
- `load_pretrain_data`: Main function to load, preprocess, align, and
  concatenate data from multiple subjects.
- `TEPParadigmWithAblation` handles the specifics of data
  extraction for TMS-EEG data types.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from moabb.datasets.base import BaseDataset
from torch.utils.data import Dataset
from TMS_EEG_moabb import (
    TEPDataset,
    TEPParadigm,
    TMSEEGClassificationTEPfree,
    TMSEEGDataset,
    TMSEEGDatasetTEPfree,
)
from tta_wrapper import (
    PYRIEMANN_AVAILABLE,
    _apply_alignment_transform_np,
    _compute_alignment_transform_np,
    _compute_reference_covariance_np,
    _compute_trial_covariances_np,
)

# --- Module-level Logger Setup ---
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


# %%
# ----------------------------------------------------------------------------
# PARADIGM AND DATASET CONFIGURATION
# ----------------------------------------------------------------------------

PARADIGM_DATA = {
    "CUSTOM_CLS": {
        "datasets": [
            "TEP",
        ],
        "class_map": {
            "TEP": TEPDataset,
        },
        "specs": {
            "TEP": dict(sr=1000, sec=0.995, n_cls=2),
        },
    },
}

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
    subjects_to_load = [s for s in subject_ids if s in available_subjects]
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
    global_backrot_np = None
    do_backrot = getattr(args, "ea_backrotation", False)

    # 1. Compute global back-rotation matrix if requested
    if do_backrot:
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
            global_backrot_np = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

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

        if do_backrot and global_backrot_np is not None:
            aligned_subject_epochs = _apply_alignment_transform_np(
                aligned_subject_epochs, global_backrot_np
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
        global_backrot_np,
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


# Backward-compatible aliases
CachingTMSEEGClassificationTEPfree = TEPParadigmWithAblation
TMSEEGClassificationTEPfreeParadigm = TEPParadigmWithAblation


# %%
# MODULE EXPORTS

__all__ = [
    "EEGDataset",
    "get_subject_list",
    "load_pretrain_data",
    "TEPParadigmWithAblation",
    "PARADIGM_DATA",
]
# %%
