#%%
"""
OnlinePredictor: an abstraction that encapsulates trial-by-trial prediction
and online finetuning for EEG classification models.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from prime.models.builder import build_model
from prime.prime_config import epoch_n_times, get_pre_epoch_time_range, get_processed_sfreq
from prime.tta_wrapper import TTAWrapper, _apply_alignment_transform_np

log = logging.getLogger(__name__)

# Model/data constants (pre-stim window from configs/prime.yaml)
_N_CHANNELS = 60          # len(COMMON_CHANNELS) after preprocessing
_PROCESSED_SFREQ = get_processed_sfreq()
_PRE_EPOCH_TMIN, _PRE_EPOCH_TMAX = get_pre_epoch_time_range()
_N_TIMEPOINTS = epoch_n_times(_PRE_EPOCH_TMIN, _PRE_EPOCH_TMAX, _PROCESSED_SFREQ)

_DEFAULT_ARGS = OmegaConf.create({
    "pre_epoch_tmin": _PRE_EPOCH_TMIN,
    "pre_epoch_tmax": _PRE_EPOCH_TMAX,
    "use_tta": True,
    "alignment_type": "euclidean",
    "use_backrotation": True,
    "alignment_ref_ema_beta": 0.99,
    "alignment_cov_epsilon": 1e-6,
    "alignment_transform_epsilon": 1e-7,
    "tta_cov_buffer_size": 50,
    "finetune_mode": "full",
    "finetune_epochs": 1,
    "batch_size_finetune": 50,
    "lr_finetune": 0.0001,
    "weight_decay_finetune": 0.0,
    "lr_calibration": 0.0001,
    "calibration_epochs": 50,
    "window_size": 100,
    "seed": 42,
})


def score_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    is_extreme_mask: Optional[np.ndarray] = None,
    original_soft_labels: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute classification metrics from predicted probabilities and labels.

    This is the single scoring function used by all evaluation stages
    (pre-calib, post-calib, online) to ensure numeric comparability.

    Args:
        predictions: Predicted probabilities, shape (n_trials,).
        labels: Ground-truth soft labels, shape (n_trials,).
        is_extreme_mask: Boolean mask for extreme trials (optional).
        original_soft_labels: Original soft labels for extreme scoring
                              (defaults to labels if not provided).

    Returns:
        Dict of metric name → value.
    """
    from sklearn.metrics import roc_auc_score

    metrics: dict = {}

    if predictions is None or len(predictions) == 0:
        return {"roc_auc_all": np.nan, "roc_auc_extreme": np.nan}

    # --- All trials ---
    true_hard = (np.asarray(labels) > 0.5).astype(int)
    if len(np.unique(true_hard)) > 1:
        metrics["roc_auc_all"] = roc_auc_score(true_hard, predictions)
    else:
        metrics["roc_auc_all"] = np.nan

    # --- Extreme trials ---
    metrics["roc_auc_extreme"] = np.nan

    if is_extreme_mask is not None and np.any(is_extreme_mask):
        extreme_idx = np.where(is_extreme_mask)[0]
        if len(extreme_idx) > 1:
            ext_preds = predictions[extreme_idx]
            ext_labels = (
                original_soft_labels[extreme_idx]
                if original_soft_labels is not None
                else labels[extreme_idx]
            )
            ext_hard = (ext_labels > 0.5).astype(int)
            if len(np.unique(ext_hard)) > 1:
                metrics["roc_auc_extreme"] = roc_auc_score(ext_hard, ext_preds)

    return metrics


class OnlinePredictor:
    """
    Wraps a TTAWrapper model and provides a clean trial-by-trial interface
    for online prediction and finetuning.

    Lifecycle contract:
        A new OnlinePredictor (and its underlying TTAWrapper) must be
        constructed per subject to avoid state leakage. Alternatively,
        call reset() between subjects to clear buffers, optimizer moments,
        and trial count.  Alignment state lives in the TTAWrapper and is
        re-initialized by calibrate().

    RNG contract:
        The constructor sets all global seeds and deterministic CUDA flags.
        To reset the RNG, construct a new OnlinePredictor.

    Usage:
        predictor = OnlinePredictor(global_backrotation, model_path="pretrained.pt", seed=42)
        predictor.calibrate(cal_trials, cal_labels)
        for epoch_pre, label in zip(intervention_epochs_pre, int_labels):
            prob = predictor.predict(epoch_pre)
            loss = predictor.finetune(epoch_pre, label)
    """

    def __init__(
        self,
        global_backrotation: np.ndarray,
        model_path: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

        self.device = torch.device("cuda")
        self.args = OmegaConf.merge(_DEFAULT_ARGS, OmegaConf.create({"seed": seed}))
        self.global_backrotation = global_backrotation

        # Build model and wrap with TTA
        base_model = build_model(
            model_name="PRIME",
            n_channels=_N_CHANNELS,
            n_times=_N_TIMEPOINTS,
            n_outputs=1,
            device=self.device,
            model_specific_args={},
        )
        self.model = TTAWrapper(
            base_model, self.args, sr_hz=_PROCESSED_SFREQ,
            global_backrotation=global_backrotation,
        ).to(self.device)

        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.wrapped_model.load_state_dict(checkpoint["model_state_dict"])

        # Setup finetuning optimizer and history buffers
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._epoch_buffer: Optional[deque] = None
        self._label_buffer: Optional[deque] = None
        self._trial_count = 0

        self._finetuning_enabled = (
            self.args.finetune_mode != "none" and self.args.finetune_epochs > 0
        )
        if self._finetuning_enabled:
            self._init_optimizer_and_buffers()

    def predict(self, epoch_pre: np.ndarray) -> float:
        """
        Predict the probability for a single trial without side effects.

        Args:
            epoch_pre: Preprocessed pre-stimulus trial as a NumPy array
                      with shape (n_channels, n_times).

        Returns:
            Predicted probability (float in [0, 1]).
        """
        self.model.eval()
        epoch_t = torch.from_numpy(epoch_pre).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model.predict(epoch_t)
            prob = torch.sigmoid(logits)
        return prob.item()

    def predict_logits(self, epoch_pre: np.ndarray) -> float:
        """
        Return raw logit for a single trial (no sigmoid).

        Useful when callers need the unnormalized output, e.g. for custom
        thresholding or external evaluation functions.
        """
        self.model.eval()
        epoch_t = torch.from_numpy(epoch_pre).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model.predict(epoch_t)
        return logits.item()

    def predict_batch(self, epochs_pre: np.ndarray, batch_size: int = 50) -> np.ndarray:
        """
        Predict probabilities for a batch of trials.

        Uses the same code path as predict() (model.predict → sigmoid),
        ensuring numeric equivalence between batch evaluation and the
        per-trial online loop.

        Args:
            epochs_pre: Preprocessed pre-stimulus trials as a NumPy array
                        with shape (n_trials, n_channels, n_times).
            batch_size: Number of trials per forward pass.

        Returns:
            Array of predicted probabilities, shape (n_trials,).
        """
        return self._predict_batch_numpy(epochs_pre, batch_size)

    def _predict_batch_numpy(self, epochs_np: np.ndarray, batch_size: int = 50) -> np.ndarray:
        """Predict probabilities from numpy epochs (n_trials, n_channels, n_times)."""
        self.model.eval()
        all_probs = []
        n_trials = len(epochs_np)
        with torch.no_grad():
            for start in range(0, n_trials, batch_size):
                batch_np = epochs_np[start : start + batch_size]
                batch_t = torch.from_numpy(batch_np).float().to(self.device)
                logits = self.model.predict(batch_t)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy().ravel())
        return np.concatenate(all_probs)

    def finetune(self, epoch_pre: np.ndarray, label: float) -> Optional[float]:
        """
        Adapt alignment (unsupervised) and perform buffered supervised
        finetuning on recent trials.

        Args:
            epoch_pre: Preprocessed pre-stimulus trial as a NumPy array
                      with shape (n_channels, n_times).
            label: The ground-truth label for this trial.

        Returns:
            The average finetuning loss if a training step was performed,
            None otherwise.
        """
        # 1. Unsupervised alignment adaptation
        if self.args.use_tta and self.args.alignment_type not in ("none", None):
            self.model.adapt_alignment(epoch_pre)

        # 2. Add to finetuning buffer
        if self._optimizer is None:
            return None

        self._epoch_buffer.append(epoch_pre)
        self._label_buffer.append(label)
        self._trial_count += 1

        # 3. Finetune when buffer has enough samples.
        # NOTE: with default config (window_size=50, batch_size_finetune=50),
        # the buffer fires exactly when full.  If these diverge, the first
        # finetuning step triggers at batch_size_finetune trials — not when
        # the buffer is full.
        min_buffer = self.args.batch_size_finetune
        if len(self._epoch_buffer) < min_buffer:
            return None

        return self._run_finetuning_step()

    def reset(self) -> None:
        """
        Reset all mutable state for reuse on a new subject.

        Clears the epoch/label buffers, trial count, and reinitializes the
        finetuning optimizer (discarding Adam moments).  Alignment state
        in the TTAWrapper is NOT reset here — call calibrate() on the new
        subject's data to reinitialize it.
        """
        self._trial_count = 0
        if self._finetuning_enabled:
            self._init_optimizer_and_buffers()

    def calibrate(
        self, cal_epochs_pre: np.ndarray, cal_labels: np.ndarray, **kwargs
    ) -> None:
        """
        Perform initial calibration: initialize alignment from calibration
        data and do supervised training over multiple epochs.

        Note: calibrate uses a throwaway optimizer (separate from the online
        finetune optimizer) whose Adam moments are intentionally discarded.
        This matches the original training path.

        Args:
            cal_epochs_pre: Preprocessed calibration trials as a NumPy array
                            with shape (n_trials, n_channels, n_times).
            cal_labels: Calibration labels, shape (n_trials,).
            **kwargs: Optional overrides for lr (lr_calibration) and
                      n_epochs (calibration_epochs).
        """
        epochs_np = cal_epochs_pre
        lr = kwargs.get("lr", getattr(self.args, "lr_calibration", 0.0001))
        n_epochs = kwargs.get(
            "n_epochs", getattr(self.args, "calibration_epochs", 50)
        )
        batch_size = min(
            self.args.batch_size_finetune, len(epochs_np)
        )

        # Initialize alignment from calibration data
        epochs_for_training = epochs_np
        if self.args.use_tta and self.args.alignment_type not in ("none", None):
            self.model.init_alignment_from_calibration(epochs_np)
            transform_np = self.model.alignment_transform_torch.cpu().numpy()
            epochs_for_training = _apply_alignment_transform_np(
                epochs_np, transform_np
            )
            if self.global_backrotation is not None:
                epochs_for_training = _apply_alignment_transform_np(
                    epochs_for_training, self.global_backrotation
                )

        # Temporarily enable full model update if in decision_only mode
        is_decision_only = (
            getattr(self.args, "finetune_mode", "full") == "decision_only"
        )
        if is_decision_only:
            self.model.enable_full_model_update(enabled=True)

        try:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr
            )
            criterion = nn.BCEWithLogitsLoss()

            cal_gen = torch.Generator()
            cal_gen.manual_seed(self.args.seed)
            loader = self._make_loader(
                epochs_for_training, cal_labels, batch_size, shuffle=True,
                generator=cal_gen,
            )
            if loader is None:
                return

            self.model.train()
            for epoch_idx in range(n_epochs):
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device).unsqueeze(1)
                    optimizer.zero_grad()
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
            self.model.eval()
        finally:
            if is_decision_only:
                self.model.enable_full_model_update(enabled=False)

    def _run_finetuning_step(self) -> float:
        """Run one finetuning step on the current buffer contents."""
        epochs_np = np.array(self._epoch_buffer)
        labels_np = np.array(self._label_buffer)

        # Apply alignment transform to buffered epochs
        if self.args.use_tta and self.args.alignment_type not in ("none", None):
            transform = self.model.alignment_transform_torch.cpu().numpy()
            epochs_np = _apply_alignment_transform_np(epochs_np, transform)
            if self.global_backrotation is not None:
                epochs_np = _apply_alignment_transform_np(
                    epochs_np, self.global_backrotation
                )

        batch_size = min(self.args.batch_size_finetune, len(epochs_np))
        finetune_gen = torch.Generator()
        finetune_gen.manual_seed(self.args.seed + self._trial_count)
        loader = self._make_loader(
            epochs_np, labels_np, batch_size, shuffle=True, generator=finetune_gen
        )
        if loader is None:
            return 0.0

        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        total_loss = 0.0

        for _ in range(self.args.finetune_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True).unsqueeze(1)
                self._optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_x, is_finetuning_batch=True)
                loss = criterion(logits, batch_y)
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()

        self.model.eval()
        avg_loss = total_loss / (len(loader) * self.args.finetune_epochs)
        return avg_loss

    def _init_optimizer_and_buffers(self) -> None:
        """(Re)initialize the finetuning optimizer and sliding-window buffers."""
        optimizer_params = {
            "lr": self.args.lr_finetune,
            "weight_decay": self.args.weight_decay_finetune,
        }
        self._optimizer = torch.optim.AdamW(
            self.model.parameters(), **optimizer_params
        )
        window_size = self.args.window_size
        self._epoch_buffer = deque(maxlen=window_size)
        self._label_buffer = deque(maxlen=window_size)

    @staticmethod
    def _make_loader(
        epochs_np: np.ndarray,
        labels_np: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Optional[DataLoader]:
        """Create a DataLoader from numpy arrays."""
        if epochs_np is None or len(epochs_np) == 0:
            return None
        epochs_t = torch.from_numpy(epochs_np).float()
        labels_t = torch.from_numpy(labels_np).float()
        dataset = TensorDataset(epochs_t, labels_t)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            generator=generator,
        )
