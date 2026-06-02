# %%
"""
CrossValidator module for managing cross-validation with separated train and test stages.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rich.console import Console
from sklearn.model_selection import KFold
from torchinfo import summary
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from prime.datasets import (
    TEPDataset,
    TEPParadigm,
    get_subject_list,
    load_pretrain_data,
)
from prime.models.builder import build_model
from prime.online_predictor import OnlinePredictor, score_predictions
from prime.utils import (
    RegressionMetricsTracker,
    filter_args_for_model,
    get_checkpoint_dir,
    get_model_class,
    save_checkpoint,
)

log = logging.getLogger(__name__)


def log_memory_usage(stage: str, log_obj=None):
    """Log current memory usage for debugging."""
    import os
    import psutil
    if log_obj:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        log_obj.debug(f"Memory usage at {stage}: {memory_mb:.1f} MB")


def create_dataloader(epochs: np.ndarray, labels: np.ndarray, batch_size: int,
                      shuffle_data: bool = True,
                      generator: torch.Generator = None) -> Optional[DataLoader]:
    """Creates a PyTorch DataLoader from NumPy arrays of epochs and labels."""
    from torch.utils.data import Dataset

    if epochs is None or labels is None or epochs.size == 0:
        return None

    class _DictDataset(Dataset):
        def __init__(self, epochs_tensor, labels_tensor):
            self.epochs = epochs_tensor
            self.labels = labels_tensor.float()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {"epoch": self.epochs[index], "label": self.labels[index]}

    epochs_tensor = torch.from_numpy(epochs).float()
    labels_tensor = torch.from_numpy(labels).float()
    dataset = _DictDataset(epochs_tensor, labels_tensor)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_data,
        num_workers=0, pin_memory=True, generator=generator,
    )


def pretrain_model(model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, n_epochs: int,
                   device: torch.device, run_name_suffix: str = "") -> nn.Module:
    """Pre-trains a model on a given dataset."""
    model.to(device)
    model.train()
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(n_epochs):
        total_epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}/{n_epochs}", leave=False)

        for batch in pbar:
            X_batch = batch['epoch'].to(device)
            y_batch = batch['label'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_epoch_loss / len(train_loader)
        log.info(f"Pre-train Epoch {epoch+1}/{n_epochs} ({run_name_suffix}) | Avg Loss: {avg_loss:.4f}")

    return model


def _run_online_finetuning(predictor, test_epochs_pre,
                           labels_for_finetuning, labels_for_evaluation,
                           is_extreme_mask, original_soft_labels,
                           args, device, console,
                           run_output_dir, subject_id, fold_idx):
    """Run an online finetuning simulation using an already-constructed OnlinePredictor."""
    n_trials = len(test_epochs_pre)
    if fold_idx is not None:
        log_prefix = f"Fold_{fold_idx}_Subj_{subject_id}_PRIME"
    else:
        log_prefix = f"Subj_{subject_id}_PRIME"
    log.info(f"Starting online simulation for {log_prefix} ({n_trials} trials)")

    metrics_tracker = RegressionMetricsTracker(window_size=args.window_size)
    trial_times = []
    trial_metrics_log = []

    online_iterator = tqdm(range(n_trials), desc=f"Online Sim ({log_prefix})", leave=False)
    for trial_idx in online_iterator:
        trial_start_time = time.time()
        epoch_pre = test_epochs_pre[trial_idx]
        single_label_for_finetuning = labels_for_finetuning[trial_idx]
        single_label_for_evaluation = labels_for_evaluation[trial_idx]

        pred_prob = predictor.predict(epoch_pre)
        step_loss_or_none = predictor.finetune(epoch_pre, single_label_for_finetuning)
        step_loss = step_loss_or_none if step_loss_or_none is not None else np.nan

        metrics_tracker.update(y_true=single_label_for_evaluation, y_pred=pred_prob)
        trial_times.append(time.time() - trial_start_time)
        online_iterator.set_postfix(auc=f"{metrics_tracker.get_rolling_roc_auc():.3f}")

        trial_metrics_log.append({
            'trial_idx': trial_idx,
            'rolling_roc_auc': metrics_tracker.get_rolling_roc_auc(),
            'overall_roc_auc_at_trial': metrics_tracker.get_overall_roc_auc(),
            'finetune_loss': step_loss,
        })

    # --- Final Metrics ---
    avg_time_per_trial = np.mean(trial_times) if trial_times else 0.0
    y_true_all = np.array(metrics_tracker.all_y_true)
    y_pred_all = np.array(metrics_tracker.all_y_pred)

    final_metrics = score_predictions(
        predictions=y_pred_all, labels=y_true_all,
        is_extreme_mask=is_extreme_mask, original_soft_labels=original_soft_labels,
    )
    log.info(f"Online sim finished. ROC AUC (All): {final_metrics.get('roc_auc_all', np.nan):.4f}.")

    if args.get('save_predictions_and_labels', False):
        output_filename = run_output_dir / f"predictions_subj_{subject_id}.npz"
        np.savez_compressed(output_filename, predictions=y_pred_all, actual_values=y_true_all)
        log.info(f"Saved predictions for Subj {subject_id} to {output_filename}")

    del metrics_tracker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_metrics, avg_time_per_trial, trial_metrics_log


# %%
# --- CrossValidator ---

class CrossValidator:
    """Manages cross-validation with clearly separated train and test stages.

    Usage:
        cv = CrossValidator(args, device, console, run_output_dir, is_cv=True)
        cv.train(train_epochs, train_labels, fold_idx=0)
        results = cv.test(test_epochs, test_labels, metadata, subject_id=101, fold_idx=0)

        # Non-CV usage:
        cv = CrossValidator(args, device, console, run_output_dir)
        cv.train(train_epochs, train_labels)
        results = cv.test(test_epochs, test_labels, metadata, subject_id=101)
    """

    def __init__(self, args: OmegaConf, device: torch.device,
                 console: Console, run_output_dir: Path, is_cv: bool = False):
        self.args = args
        self.device = device
        self.console = console
        self.run_output_dir = run_output_dir
        self.is_cv = is_cv
        self.model_path: Optional[Path] = None
        self.n_channels: int = -1
        self.n_timepoints: int = -1
        self.global_backrotation: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE: TRAIN
    # ------------------------------------------------------------------

    def train(self, epochs: np.ndarray, labels: np.ndarray,
              fold_idx: Optional[int] = None) -> None:
        """Train the PRIME model on the provided data.

        Args:
            epochs: Training EEG data with shape (n_trials, n_channels, n_times).
            labels: Training labels with shape (n_trials,).
            fold_idx: Fold index (required when is_cv=True).
        """
        if self.is_cv:
            assert fold_idx is not None, "fold_idx is required in CV mode."
        log_memory_usage(f"start_training_fold_{fold_idx}" if fold_idx is not None else "start_training", log)
        self.console.print(f"  Training on {len(epochs)} trials...")

        self.n_channels = epochs.shape[1]
        self.n_timepoints = epochs.shape[2]

        pretrain_gen = torch.Generator()
        pretrain_gen.manual_seed(self.args.seed)
        train_loader = create_dataloader(
            epochs, labels, self.args.batch_size_pretrain,
            shuffle_data=True, generator=pretrain_gen,
        )
        assert train_loader is not None and len(train_loader) > 0, \
            "Could not create a valid dataloader from training data."

        self.console.print(f"    Training model: [bold yellow]PRIME[/bold yellow]")
        base_args_dict = OmegaConf.to_container(self.args, resolve=True)
        model_specific_args = filter_args_for_model(
            base_args_dict, "PRIME", get_model_class("PRIME")
        )
        model = build_model(
            model_name="PRIME",
            n_channels=self.n_channels,
            n_times=self.n_timepoints,
            n_outputs=1,
            device=self.device,
            model_specific_args=model_specific_args,
            target_type="classification",
        )

        optimizer_params = {
            "lr": self.args.lr_pretrain,
            "weight_decay": self.args.weight_decay_pretrain,
        }
        if self.args.optimizer_type_pretrain.lower() == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        else:
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

        summary_str = summary(model, input_size=(1, self.n_channels, self.n_timepoints), verbose=0)
        self.console.print(str(summary_str))

        model = pretrain_model(
            model=model, train_loader=train_loader, optimizer=optimizer,
            n_epochs=self.args.pretrain_epochs, device=self.device,
            run_name_suffix=f"Fold_{fold_idx+1}_PRIME" if fold_idx is not None else "PRIME",
        )

        # Always save pretrained model so OnlinePredictor can load from path
        if self.is_cv:
            suffix = f"_fold_{fold_idx+1}"
        elif self.args.get("train_all", False):
            suffix = "_all"
        else:
            suffix = ""
        save_path = self.run_output_dir / f"pretrained{suffix}.pt"
        save_checkpoint({"model_state_dict": model.state_dict()}, save_path)
        self.model_path = save_path
        self.console.print(f"      [green]Saved pretrained model to {save_path.name}[/green]")

        if self.args.save_checkpoints:
            checkpoint_dir = get_checkpoint_dir(self.run_output_dir)
            label = f"model_PRIME_fold_{fold_idx+1}_pretrained.pt" if fold_idx is not None else "model_PRIME_pretrained.pt"
            checkpoint_save_path = checkpoint_dir / label
            save_checkpoint({"model_state_dict": model.state_dict()}, checkpoint_save_path)

        # Save global back-rotation matrix if produced
        if self.global_backrotation is not None:
            if self.is_cv:
                suffix = f"_fold_{fold_idx+1}"
            elif self.args.get("train_all", False):
                suffix = "_all"
            else:
                suffix = ""
            backrotation_path = self.run_output_dir / f"global_backrotation{suffix}.npy"
            np.save(backrotation_path, self.global_backrotation)
            self.console.print(f"    [green]Saved global back-rotation matrix to {backrotation_path.name}[/green]")

        del model, optimizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.console.print(f"    [green]Training complete{f' for fold {fold_idx+1}' if fold_idx is not None else ''}.[/green]")

    def load_pretrained(self, checkpoint_dir: Path, fold_idx: Optional[int] = None) -> None:
        """Load pretrained model weights from a checkpoint directory.

        Args:
            checkpoint_dir: Directory containing pretrained checkpoint.
            fold_idx: Fold index (required when is_cv=True).
        """
        if self.is_cv:
            assert fold_idx is not None, "fold_idx is required in CV mode."
        checkpoint_dir = Path(checkpoint_dir)
        if self.is_cv:
            chkpt_path = checkpoint_dir / f"pretrained_fold_{fold_idx+1}.pt"
        else:
            chkpt_path = checkpoint_dir / "pretrained.pt"
        assert chkpt_path.is_file(), f"Checkpoint not found: {chkpt_path}"

        self.model_path = chkpt_path
        self.console.print(f"  Loaded checkpoint: {chkpt_path}")

        # Load back-rotation matrix if available
        if getattr(self.args, "use_backrotation", False):
            if self.is_cv:
                backrotation_path = checkpoint_dir / f"global_backrotation_fold_{fold_idx+1}.npy"
            else:
                backrotation_path = checkpoint_dir / "global_backrotation.npy"
            assert backrotation_path.exists(), f"Back-rotation matrix not found: {backrotation_path}"
            self.global_backrotation = np.load(backrotation_path)
            self.console.print(f"  [green]Loaded global back-rotation matrix from {backrotation_path.name}.[/green]")

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE: TEST
    # ------------------------------------------------------------------

    def test(self, epochs: np.ndarray, labels: np.ndarray,
             metadata: Optional[pd.DataFrame] = None,
             subject_id: int = 0, fold_idx: Optional[int] = None) -> Tuple[dict, list]:
        """Test the trained model on the provided data.

        Runs pre-calibration evaluation, optional calibration, and online
        fine-tuning simulation.

        Args:
            epochs: Test EEG data with shape (n_trials, n_channels, n_times).
            labels: Test labels (soft/ground truth) with shape (n_trials,).
            metadata: DataFrame with 'period' column for calibration/intervention split.
            subject_id: Subject identifier for logging and file naming.
            fold_idx: Fold index (required when is_cv=True).

        Returns:
            Tuple of (stage_results, per_trial_metrics).
        """
        if self.is_cv:
            assert fold_idx is not None, "fold_idx is required in CV mode."
            self.console.print(f"  Testing Subject {subject_id} (Fold {fold_idx+1})...")
        else:
            self.console.print(f"  Testing Subject {subject_id}...")
        assert epochs is not None and epochs.size > 0, \
            f"No valid data for subject {subject_id}."

        if self.n_channels == -1 or self.n_timepoints == -1:
            self.n_channels = epochs.shape[1]
            self.n_timepoints = epochs.shape[2]

        # Load back-rotation matrix if needed and not already loaded
        if getattr(self.args, "use_backrotation", False) and self.global_backrotation is None:
            if self.is_cv:
                filename = f"global_backrotation_fold_{fold_idx+1}.npy"
            else:
                filename = "global_backrotation.npy"
            # Determine search directory
            if getattr(self.args, "pretrained_checkpoint_dir", None):
                search_dir = Path(self.args.pretrained_checkpoint_dir)
            else:
                search_dir = self.run_output_dir
            backrotation_path = search_dir / filename
            assert backrotation_path.exists(), \
                f"Back-rotation is ON but file not found: {backrotation_path}"
            self.global_backrotation = np.load(backrotation_path)

        # --- Label Preparation ---
        is_extreme_mask = (labels <= 0.25) | (labels >= 0.75)
        labels_ground_truth = labels

        if getattr(self.args, "shuffle_test_labels", False):
            self.console.print("[bold red]WARNING: SHUFFLING TEST LABELS FOR CONTROL ANALYSIS.[/bold red]")
            from sklearn.utils import shuffle
            labels_for_eval = shuffle(labels_ground_truth.copy(), random_state=self.args.seed + subject_id)
        else:
            labels_for_eval = labels_ground_truth

        # --- Build predictor ---
        predictor = OnlinePredictor(
            self.global_backrotation, model_path=self.model_path, seed=self.args.seed,
        )
        if self.model_path is not None:
            self.console.print("        Loaded pre-trained state.")

        stage_results = {"pre_calib_zero_shot": {}, "post_calib_zero_shot": {}, "finetuned": {}}

        # --- STAGE 1: PRE-CALIBRATION EVALUATION ---
        self.console.print(f"      Pre-Calibration Zero-Shot on {len(epochs)} trials...")
        pre_calib_preds = predictor.predict_batch(epochs, batch_size=self.args.batch_size_finetune)
        pre_calib_metrics = score_predictions(
            predictions=pre_calib_preds, labels=labels_for_eval,
            is_extreme_mask=is_extreme_mask, original_soft_labels=labels_for_eval,
        )
        stage_results["pre_calib_zero_shot"] = pre_calib_metrics
        self.console.print(f"      [bold]Pre-Calib ROC AUC: {pre_calib_metrics.get('roc_auc_all', np.nan):.4f}[/bold]")

        # --- Split data for calibration and online phases ---
        if metadata is not None and 'period' in metadata.columns:
            cal_mask = (metadata['period'] == 'calibration').values
            int_mask = (metadata['period'] == 'intervention').values
            calibration_epochs_pre = epochs[cal_mask]
            online_epochs_pre = epochs[int_mask]
            calibration_labels = labels_ground_truth[cal_mask]
            online_labels_for_finetuning = labels_ground_truth[int_mask]
            online_labels_for_eval = labels_for_eval[int_mask]
            online_is_extreme_mask = is_extreme_mask[int_mask]
        else:
            calibration_epochs_pre = None
            online_epochs_pre = epochs
            online_labels_for_finetuning = labels_ground_truth
            online_labels_for_eval = labels_for_eval
            online_is_extreme_mask = is_extreme_mask

        # --- STAGE 2: CALIBRATION ---
        if calibration_epochs_pre is not None and len(calibration_epochs_pre) > 0:
            self.console.print(f"      Calibrating for {self.args.calibration_epochs} epochs...")
            predictor.calibrate(calibration_epochs_pre, calibration_labels)
            self.console.print("      [green]Calibration complete.[/green]")

        # --- STAGE 3: ONLINE EVALUATION ---
        per_trial_metrics = []
        if online_epochs_pre is not None and len(online_epochs_pre) > 0:
            self.console.print(f"      Post-Calibration Zero-Shot on {len(online_epochs_pre)} trials...")
            post_calib_preds = predictor.predict_batch(online_epochs_pre, batch_size=self.args.batch_size_finetune)
            post_calib_metrics = score_predictions(
                predictions=post_calib_preds, labels=online_labels_for_eval,
                is_extreme_mask=online_is_extreme_mask, original_soft_labels=online_labels_for_eval,
            )
            stage_results["post_calib_zero_shot"] = post_calib_metrics
            self.console.print(f"      [bold]Post-Calib ROC AUC: {post_calib_metrics.get('roc_auc_all', np.nan):.4f}[/bold]")

            self.console.print(f"      Online finetuning on {len(online_epochs_pre)} trials...")
            final_finetuned_metrics, _, per_trial_metrics = _run_online_finetuning(
                predictor=predictor,
                test_epochs_pre=online_epochs_pre,
                labels_for_finetuning=online_labels_for_finetuning,
                labels_for_evaluation=online_labels_for_eval,
                is_extreme_mask=online_is_extreme_mask,
                original_soft_labels=online_labels_for_eval,
                args=self.args, device=self.device, console=self.console,
                run_output_dir=self.run_output_dir,
                subject_id=subject_id, fold_idx=fold_idx + 1 if fold_idx is not None else None,
            )
            stage_results["finetuned"] = final_finetuned_metrics

            if self.args.get('save_finetuned_model', False):
                save_path = self.run_output_dir / f"finetuned_subj_{subject_id}.pt"
                save_checkpoint({'model_state_dict': predictor.model.state_dict()}, save_path)
                self.console.print(f"      [green]Saved fine-tuned model to {save_path.name}[/green]")
        else:
            self.console.print("      No data available for the online phase.")

        return stage_results, per_trial_metrics

    # ------------------------------------------------------------------
    # CONVENIENCE: FULL K-FOLD PIPELINE
    # ------------------------------------------------------------------

    def run_kfold(self) -> Tuple[dict, list]:
        """Execute the full k-fold cross-validation pipeline.

        Returns:
            Tuple of (results_per_fold, all_trial_metrics).
        """
        all_subjects = get_subject_list(self.args.data_root)
        subjects_to_run = (
            [s for s in all_subjects if s in self.args.subjects]
            if self.args.subjects else all_subjects
        )
        assert len(subjects_to_run) >= self.args.n_splits, \
            f"Insufficient subjects ({len(subjects_to_run)}) for {self.args.n_splits} splits."

        kf = KFold(n_splits=self.args.n_splits, shuffle=True, random_state=self.args.seed)
        fold_results = {f: {} for f in range(self.args.n_splits)}
        all_trial_metrics = []

        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(subjects_to_run)):
            train_subject_ids = [subjects_to_run[i] for i in train_indices]
            test_subject_ids = [subjects_to_run[i] for i in test_indices]

            run_only_fold = getattr(self.args, "run_only_fold", None)
            if run_only_fold is not None and (fold_idx + 1) != run_only_fold:
                continue

            self.console.print(
                f"\n  [bold blue]=> Fold {fold_idx+1}/{self.args.n_splits} "
                f"| Train: {train_subject_ids} | Test: {test_subject_ids}[/bold blue]"
            )

            # --- TRAINING STAGE ---
            self._prepare_fold(fold_idx, train_subject_ids)

            # --- TESTING STAGE ---
            max_test_subjs = getattr(self.args, "max_test_subjects_per_fold", None)
            for subj_count, test_subject_id in enumerate(test_subject_ids):
                test_epochs, test_labels, test_metadata = self._load_test_subject_data(test_subject_id)
                subject_results, subject_trial_metrics = self.test(
                    epochs=test_epochs, labels=test_labels,
                    metadata=test_metadata,
                    subject_id=test_subject_id, fold_idx=fold_idx,
                )

                fold_results[fold_idx][test_subject_id] = subject_results
                for entry in subject_trial_metrics:
                    entry.update({"fold": fold_idx + 1, "subject_id": test_subject_id})
                    all_trial_metrics.append(entry)

                if max_test_subjs is not None and (subj_count + 1) >= max_test_subjs:
                    self.console.print(
                        f"  [bold yellow]Reached max_test_subjects_per_fold={max_test_subjs}. Stopping.[/bold yellow]"
                    )
                    break

        return fold_results, all_trial_metrics

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _prepare_fold(self, fold_idx: int, train_subject_ids: list) -> None:
        """Load training data and train, or load pretrained checkpoint."""
        if self.args.pretrained_checkpoint_dir:
            log.info(f"Loading pre-trained model from: {self.args.pretrained_checkpoint_dir}")
            _, _, n_ch, n_tp, _, _ = load_pretrain_data(
                subject_ids=[train_subject_ids[0]],
                paradigm_kwargs=self._paradigm_kwargs(),
                data_root=self.args.data_root, args=self.args,
            )
            self.n_channels, self.n_timepoints = n_ch, n_tp
            self.load_pretrained(Path(self.args.pretrained_checkpoint_dir), fold_idx)

        elif not self.args.no_pretrain:
            train_epochs, train_labels = self._load_train_data(train_subject_ids)
            assert train_epochs is not None and train_epochs.size > 0, \
                f"No training data loaded for fold {fold_idx+1}."
            self.train(train_epochs, train_labels, fold_idx=fold_idx)

        else:
            log.info("`no_pretrain` is True. Model will be randomly initialized.")
            _, _, n_ch, n_tp, _, _ = load_pretrain_data(
                subject_ids=[train_subject_ids[0]],
                paradigm_kwargs=self._paradigm_kwargs(),
                data_root=self.args.data_root, args=self.args,
            )
            self.n_channels, self.n_timepoints = n_ch, n_tp
            assert n_ch > 0 and n_tp > 0, "Could not determine data dimensions."

    def _paradigm_kwargs(self) -> dict:
        return {
            "fmin": self.args.fmin, "fmax": self.args.fmax,
            "resample": self.args.resample,
        }

    def _load_train_data(self, train_subject_ids: list) -> Tuple[np.ndarray, np.ndarray]:
        """Load and return training data for the given subjects."""
        actual_subject_ids = list(train_subject_ids)
        self.console.print(f"  Loading pretraining data for {len(actual_subject_ids)} subjects...")

        paradigm_kwargs = self._paradigm_kwargs()
        if hasattr(self.args, "channel_subset") and self.args.channel_subset:
            paradigm_kwargs["channels"] = self.args.channel_subset

        epochs_data, labels_data, _, _, _, global_backrotation = load_pretrain_data(
            subject_ids=actual_subject_ids,
            paradigm_kwargs=paradigm_kwargs,
            data_root=self.args.data_root,
            args=self.args,
            apply_trial_ablation=True,
        )

        if global_backrotation is not None:
            self.global_backrotation = global_backrotation

        assert epochs_data is not None and epochs_data.size > 0, "No pretraining data loaded."
        self.console.print(f"    Total pretrain trials: {len(epochs_data)}.")
        return epochs_data, labels_data

    def _load_test_subject_data(self, test_subject_id: int
                                ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Load test data for a single subject."""
        dataset = TEPDataset(data_path=self.args.data_root)
        paradigm = TEPParadigm()
        test_epochs, test_labels, test_metadata = paradigm.get_data(
            dataset=dataset, subjects=[test_subject_id]
        )
        return test_epochs, test_labels, test_metadata
