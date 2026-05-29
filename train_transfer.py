#%%
"""
Main script for conducting transfer learning experiments on EEG-TMS data using 
cross-subject k-fold cross-validation.

This script handles:
- Pre-training models on a set of source subjects.
- Evaluating models in a zero-shot manner on a target subject.
- Performing subject-specific calibration.
- Simulating online fine-tuning on a trial-by-trial basis.
- Aggregating and reporting performance metrics.
"""

# Core Python libraries
import argparse
import copy
import gc
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil

# Third-party libraries for data handling and computation
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from omegaconf import ListConfig
from rich.table import Table


# Visualization and utility libraries
from rich.console import Console
from torchinfo import summary

# Local project-specific modules
from datasets import *
from TMS_EEG_moabb import TMSEEGDatasetTEPfree, TMSEEGClassificationTEPfree
from models.builder import build_model
from tta_wrapper import TTAWrapper
from online_predictor import OnlinePredictor, score_predictions
from utils import (RegressionMetricsTracker, filter_args_for_model,
                   get_checkpoint_dir, get_model_class,
                   get_output_dir, save_checkpoint, save_results_df)

# Optional dependencies for specific functionalities
try:
    import pyriemann
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# --- Global Setup ---
# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")

# Setup main logger
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# %% --- Data Handling ---

class DictDataset(Dataset):
    """A simple PyTorch Dataset that accepts tensors for epochs and labels."""
    def __init__(self, epochs_tensor: torch.Tensor, labels_tensor: torch.Tensor):
        if len(epochs_tensor) != len(labels_tensor):
            raise ValueError("Epochs and labels must have the same length.")
        self.epochs = epochs_tensor
        # Ensure labels are float for compatibility with regression and BCE loss
        self.labels = labels_tensor.float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"epoch": self.epochs[index], "label": self.labels[index]}


def create_dataloader(epochs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle_data: bool = True, generator: torch.Generator = None) -> Optional[DataLoader]:
    """Creates a PyTorch DataLoader from NumPy arrays of epochs and labels.
    
    Args:
        epochs: EEG data with shape (n_trials, n_channels, n_times).
        labels: Target labels with shape (n_trials,).
        batch_size: The number of samples per batch.
        shuffle_data: Whether to shuffle the data at every epoch.
        generator: Optional torch.Generator for reproducible shuffling.

    Returns:
        A DataLoader instance or None if input data is invalid.
    """
    if epochs is None or labels is None or epochs.size == 0:
        return None
        
    epochs_tensor = torch.from_numpy(epochs).float()
    labels_tensor = torch.from_numpy(labels).float()
    
    dataset = DictDataset(epochs_tensor, labels_tensor)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=0,  # Set to 0 for main process loading, adjust if needed
        pin_memory=True,
        generator=generator
    )


# %% --- Core Training Functions ---

def pretrain_model(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   n_epochs: int, device: torch.device, args: OmegaConf,
                   run_name_suffix: str = "") -> nn.Module:
    """Pre-trains a model on a given dataset.
    """
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


# %%
# Experiment setup

def setup_experiment(cli_args=None):
    """Parse arguments, setup configuration, and initialize experiment."""
    
    DEFAULT_YAML = """
dataset_names: ["TMSEEGClassificationTEPfree"]
subjects: null
data_root: "~/prime-data/processed"
pretrained_checkpoint_dir: null
fmin: null
fmax: null
tmin: -0.55
tmax: -0.050
resample: null
models_to_run:
  - ShallowConvNet
  - DeepConvNet
  - EEGNetv4
  - ATCNet
  - PRIME
  - Ablation_NoS4
  - Ablation_ConvInsteadOfS4
  - Ablation_S4_WithConvClassifier
n_splits: 2
pretrain_epochs: 100
lr_pretrain: 0.0003
optimizer_type_pretrain: "AdamW"
weight_decay_pretrain: 0.0
batch_size_pretrain: 64
window_size: 50
finetune_epochs: 1
finetune_warmup_trials: 0
lr_finetune: 0.0001
optimizer_type_finetune: "AdamW" # Options: "Adam", "AdamW"
weight_decay_finetune: 0.0
batch_size_finetune: 50 
seed: 42
device: "cuda"
print_dataset_structure_and_exit: false
no_pretrain: false
base_output_dir: "results"
save_pretrained_model: false
save_finetuned_model: false
save_checkpoints: false
save_results: true
save_predictions_and_labels: true
use_wandb: false
wandb_project: "kfold_transfer"
wandb_group_prefix: "kfold_transfer"
wandb_run_description: "Default run"
num_pretrain_subjects: "max"
num_finetuning_subjects: null
num_trials_per_subject: null


# TTA Configuration
use_tta: true
ea_backrotation: true
alignment_type: "euclidean" # Options: "none", "euclidean", "riemannian" 
alignment_cov_epsilon: 1.0e-6
alignment_transform_epsilon: 1.0e-7
alignment_ref_ema_beta: 0.99
tta_cov_buffer_size: 50
use_adabn: false
finetune_mode: "full" # Options: "full", "decision_only", "none", "decision_criterion_only"

experiment_mode: "cross_subject_kfold"  

# channel_subset:
#   - C3
#   - FC1
#   - FC5
#   - CP1
#   - CP5
#   - C4
#   - FC2
#   - FC6
#   - CP2
#   - CP6

use_subject_specific_calibration: true 
num_calibration_trials: 100       # Number of initial trials from the test subject to use for calibration
lr_calibration: 0.0001           
calibration_epochs: 50           # Number of epochs to train on the small calibration set

shuffle_test_labels: false
max_test_subjects_per_fold: null
"""
    
    # Load default config
    config = OmegaConf.create(DEFAULT_YAML)

    # Suppress verbose MOABB warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="moabb")
    warnings.filterwarnings("ignore", message="warnEpochs*")

    # Set MOABB loggers to ERROR level
    logging.getLogger("moabb").setLevel(logging.ERROR)
    logging.getLogger("moabb.paradigms").setLevel(logging.ERROR)
    logging.getLogger("moabb.datasets").setLevel(logging.ERROR)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="K-Fold Transfer with YAML Config")
    parser.add_argument("-c", "--config", action="append", help="Path to YAML config file(s)", default=[])
    parser.add_argument("--print-dataset-structure-and-exit", action="store_true")
    
    parsed_args, remaining_argv = parser.parse_known_args(args=cli_args)
    
    # Load and merge config files
    if parsed_args.config:
        for config_file in parsed_args.config:
            try:
                user_config = OmegaConf.load(config_file)
                config = OmegaConf.merge(config, user_config)
                print(f"Loaded config from: {config_file}")
            except Exception as e:
                print(f"Failed to load config {config_file}: {e}")
    
    # Apply CLI overrides
    if remaining_argv:
        try:
            cli_conf = OmegaConf.from_cli(remaining_argv)
            if cli_conf:
                config = OmegaConf.merge(config, cli_conf)
        except Exception as e:
            print(f"Failed to parse CLI overrides: {e}")
    
    if parsed_args.print_dataset_structure_and_exit:
        config.print_dataset_structure_and_exit = True
    
    OmegaConf.resolve(config)
    config.data_root = str(Path(config.data_root).expanduser())
    
    # Set seed and enforce deterministic CUDA operations
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Create output directory — results/[timestamp]_[config_name]/
    config_name = ""
    if parsed_args.config:
        config_name = Path(parsed_args.config[-1]).stem
    run_output_dir = get_output_dir(
        base_output_root=config.base_output_dir,
        config_name=config_name,
    )
    console = Console()
    console.print(f"[blue]Output directory: {run_output_dir}[/blue]")
    
    # Save config
    try:
        config_save_path = run_output_dir / "config.yaml"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, config_save_path)
    except Exception as e:
        print(f"Failed to save config: {e}")
    
    # Setup device
    device = torch.device(config.device)
    if config.device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available. Switching to CPU.[/yellow]")
        device = torch.device("cpu")
        config.device = "cpu"
    
    # Environment flags
    config.pyriemann_available = PYRIEMANN_AVAILABLE
    if config.use_wandb and not WANDB_AVAILABLE:
        console.print("[yellow]WandB not available. Disabling.[/yellow]")
        config.use_wandb = False
    config.wandb_available = WANDB_AVAILABLE
    
    # Process subjects and dataset names
    if config.subjects is not None:
        if isinstance(config.subjects, int):
            config.subjects = [config.subjects]
        elif isinstance(config.subjects, (list, ListConfig)):
            config.subjects = [int(s) for s in config.subjects]
    
    if not isinstance(config.dataset_names, (list, ListConfig)):
        if isinstance(config.dataset_names, str):
            config.dataset_names = [d.strip() for d in config.dataset_names.split(",") if d.strip()]
    config.dataset_names = [str(d) for d in config.dataset_names]
    
    return config, device, console, run_output_dir

class CrossValidator:
    """Manages cross-validation with clearly separated train and test stages.

    Usage:
        cv = CrossValidator(args, device, console, run_output_dir)

        # Train with data
        cv.train(train_epochs, train_labels, fold_idx=0)

        # Or load pretrained weights instead of training
        cv.load_pretrained(checkpoint_dir, fold_idx=0)

        # Test with data
        results = cv.test(test_epochs, test_labels, metadata, subject_id=101, fold_idx=0)

        # Or run full k-fold pipeline
        all_results = cv.run_kfold(dataset_name)
    """

    def __init__(self, args: OmegaConf, device: torch.device, console: Console, run_output_dir: Path):
        self.args = args
        self.device = device
        self.console = console
        self.run_output_dir = run_output_dir
        self.trained_models: Dict[str, dict] = {}  # model_name -> state_dict
        self.n_channels: int = -1
        self.n_timepoints: int = -1
        self.global_backrot_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE: TRAIN
    # ------------------------------------------------------------------

    def train(self, epochs: np.ndarray, labels: np.ndarray, fold_idx: int = 0,
              dataset_name: str = "") -> bool:
        """Train all configured models on the provided data.

        Args:
            epochs: Training EEG data with shape (n_trials, n_channels, n_times).
            labels: Training labels with shape (n_trials,).
            fold_idx: Fold index for logging and checkpoint saving.
            dataset_name: Name of the dataset (used for logging).

        Returns:
            True if at least one model was trained successfully.
        """
        log_memory_usage(f"start_training_fold_{fold_idx+1}", log)
        self.console.print(f"  Training on {len(epochs)} trials...")

        self.n_channels = epochs.shape[1]
        self.n_timepoints = epochs.shape[2]
        self.trained_models = {}

        # --- Create DataLoader ---
        pretrain_gen = torch.Generator()
        pretrain_gen.manual_seed(self.args.seed)
        train_loader = create_dataloader(
            epochs, labels, self.args.batch_size_pretrain,
            shuffle_data=True, generator=pretrain_gen)

        if train_loader is None or len(train_loader) == 0:
            self.console.print("[yellow]Could not create a valid dataloader. Training aborted.[/yellow]")
            return False

        # --- Train each model ---
        self.console.print(f"    Training models: {self.args.models_to_run}")
        base_args_dict = OmegaConf.to_container(self.args, resolve=True)
        printed_summaries = set()

        for model_idx, model_name in enumerate(self.args.models_to_run):
            self.console.print(
                f"      Model {model_idx+1}/{len(self.args.models_to_run)}: [bold yellow]{model_name}[/bold yellow]"
            )
            try:
                model_specific_args = filter_args_for_model(
                    base_args_dict, model_name, get_model_class(model_name)
                )
                model_obj = build_model(
                    model_name=model_name,
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
                    optimizer = torch.optim.AdamW(model_obj.parameters(), **optimizer_params)
                else:
                    optimizer = torch.optim.Adam(model_obj.parameters(), **optimizer_params)

                if (
                    (dataset_name, model_name) not in printed_summaries
                    and self.n_channels > 0
                    and self.n_timepoints > 0
                ):
                    try:
                        summary_str = summary(
                            model_obj,
                            input_size=(1, self.n_channels, self.n_timepoints),
                            verbose=0,
                        )
                        self.console.print(str(summary_str))
                        printed_summaries.add((dataset_name, model_name))
                    except Exception:
                        pass

                model_obj = pretrain_model(
                    model=model_obj,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    n_epochs=self.args.pretrain_epochs,
                    device=self.device,
                    args=self.args,
                    run_name_suffix=f"{dataset_name}_Fold_{fold_idx+1}_{model_name}",
                )

                self.trained_models[model_name] = copy.deepcopy(model_obj.state_dict())

                # Save pretrained model checkpoint
                if self.args.get("save_pretrained_model", False):
                    save_path = self.run_output_dir / f"pretrained_fold_{fold_idx+1}.pt"
                    save_checkpoint(
                        {"model_state_dict": model_obj.state_dict()}, save_path
                    )
                    self.console.print(
                        f"      [green]Saved pretrained model to {save_path.name}[/green]"
                    )

                if self.args.save_checkpoints:
                    checkpoint_dir = get_checkpoint_dir(self.run_output_dir)
                    save_path = (
                        checkpoint_dir
                        / f"model_{model_name}_ds_{dataset_name}_fold_{fold_idx+1}_pretrained.pt"
                    )
                    save_checkpoint(
                        {"model_state_dict": self.trained_models[model_name]},
                        save_path,
                    )

            except Exception as e:
                log.error(f"Error training {model_name} in fold {fold_idx+1}: {e}", exc_info=True)
                self.console.print(f"      [red]Failed to train {model_name}: {e}[/red]")
            finally:
                if "model_obj" in locals():
                    del model_obj
                if "optimizer" in locals():
                    del optimizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Save global back-rotation matrix if produced
        if self.global_backrot_matrix is not None:
            backrot_matrix_path = self.run_output_dir / f"global_backrotation_matrix_fold_{fold_idx+1}.npy"
            try:
                np.save(backrot_matrix_path, self.global_backrot_matrix)
                self.console.print(f"    [green]Saved global back-rotation matrix to {backrot_matrix_path.name}[/green]")
            except Exception as e:
                self.console.print(f"    [red]Failed to save back-rotation matrix: {e}[/red]")

        if not self.trained_models:
            self.console.print(f"[red]No models were successfully trained in fold {fold_idx+1}.[/red]")
            return False

        self.console.print(
            f"    [green]Successfully trained {len(self.trained_models)} models for fold {fold_idx+1}.[/green]"
        )
        return True

    def load_pretrained(self, checkpoint_dir: Path, fold_idx: int = 0) -> bool:
        """Load pretrained model weights from a checkpoint directory.

        Args:
            checkpoint_dir: Directory containing pretrained_fold_N.pt files.
            fold_idx: Fold index to load (0-based).

        Returns:
            True if at least one model was loaded successfully.
        """
        checkpoint_dir = Path(checkpoint_dir)
        self.trained_models = {}

        for model_name in self.args.models_to_run:
            chkpt_path = checkpoint_dir / f"pretrained_fold_{fold_idx+1}.pt"
            if chkpt_path.is_file():
                state_dict = torch.load(chkpt_path, map_location='cpu')['model_state_dict']
                self.trained_models[model_name] = state_dict
                self.console.print(f"  Loaded checkpoint: {chkpt_path}")
            else:
                log.error(f"Checkpoint not found: {chkpt_path}")

        # Load back-rotation matrix if available
        if getattr(self.args, "ea_backrotation", False):
            backrot_path = checkpoint_dir / f"global_backrotation_matrix_fold_{fold_idx+1}.npy"
            if backrot_path.exists():
                self.global_backrot_matrix = np.load(backrot_path)
                self.console.print(f"  [green]Loaded global back-rotation matrix from {checkpoint_dir}.[/green]")

        return bool(self.trained_models)

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE: TEST
    # ------------------------------------------------------------------

    def test(self, epochs: np.ndarray, labels: np.ndarray,
             metadata: Optional[pd.DataFrame] = None,
             subject_id: int = 0, fold_idx: int = 0,
             dataset_name: str = "") -> Tuple[Dict[str, dict], Dict[str, list]]:
        """Test trained models on the provided data.

        Runs pre-calibration evaluation, optional calibration, and online
        fine-tuning simulation on the provided epochs/labels.

        Args:
            epochs: Test EEG data with shape (n_trials, n_channels, n_times).
            labels: Test labels (soft/ground truth) with shape (n_trials,).
            metadata: DataFrame with 'period' column for calibration/intervention split.
                      If None, all data is treated as intervention (online phase).
            subject_id: Subject identifier for logging and file naming.
            fold_idx: Fold index for logging and file naming.
            dataset_name: Name of the dataset (used for logging).

        Returns:
            Tuple of (subject_results, trial_metrics_per_model).
            subject_results: {model_name: {stage: metrics_dict}}
            trial_metrics_per_model: {model_name: [per_trial_dicts]}
        """
        subject_results = {model_name: {} for model_name in self.args.models_to_run}
        all_models_trial_metrics = {}
        self.console.print(f"  Testing Subject {subject_id} (Dataset: {dataset_name}, Fold {fold_idx+1})...")

        if epochs is None or epochs.size == 0:
            self.console.print(f"    [yellow]No valid data for subject {subject_id}. Skipping.[/yellow]")
            return subject_results, all_models_trial_metrics

        # Update dimensions from test data if not already set
        if self.n_channels == -1 or self.n_timepoints == -1:
            self.n_channels = epochs.shape[1]
            self.n_timepoints = epochs.shape[2]

        # Load back-rotation matrix if needed and not already loaded
        if getattr(self.args, "ea_backrotation", False) and self.global_backrot_matrix is None:
            backrot_matrix_path = self.run_output_dir / f"global_backrotation_matrix_fold_{fold_idx+1}.npy"
            if backrot_matrix_path.exists():
                self.global_backrot_matrix = np.load(backrot_matrix_path)
            else:
                raise FileNotFoundError(
                    f"Back-rotation is ON but matrix file was not found at {backrot_matrix_path}."
                )

        # --- Label Preparation ---
        is_extreme_mask = (labels <= 0.25) | (labels >= 0.75)
        labels_ground_truth = labels

        if getattr(self.args, "shuffle_test_labels", False):
            self.console.print("[bold red]WARNING: SHUFFLING TEST SUBJECT LABELS FOR CONTROL ANALYSIS.[/bold red]")
            from sklearn.utils import shuffle
            labels_for_eval = shuffle(
                labels_ground_truth.copy(),
                random_state=self.args.seed + subject_id
            )
        else:
            labels_for_eval = labels_ground_truth

        sr_hz_eval = next(
            (p_data["specs"][dataset_name].get("sr")
             for p_name, p_data in PARADIGM_DATA.items()
             if dataset_name in p_data.get("specs", {})),
            None
        )

        try:
            for model_name in self.args.models_to_run:
                self.console.print(f"      Evaluating Model: [bold yellow]{model_name}[/bold yellow]")
                try:
                    subject_results[model_name] = {
                        "pre_calib_zero_shot": {},
                        "post_calib_zero_shot": {},
                        "finetuned": {},
                    }

                    model_eval = build_model(
                        model_name=model_name, n_channels=self.n_channels,
                        n_times=self.n_timepoints, n_outputs=1,
                        device=self.device,
                        model_specific_args=filter_args_for_model(
                            OmegaConf.to_container(self.args, resolve=True),
                            model_name, get_model_class(model_name)
                        )
                    )
                    model_eval_wrapped = TTAWrapper(
                        model_eval, self.args, sr_hz=sr_hz_eval,
                        global_backrot_matrix_np=self.global_backrot_matrix
                    ).to(self.device)

                    if model_name in self.trained_models:
                        model_eval_wrapped.wrapped_model.load_state_dict(self.trained_models[model_name])
                        self.console.print("        Loaded pre-trained state.")

                    predictor = OnlinePredictor(
                        model=model_eval_wrapped, args=self.args, device=self.device,
                        global_backrot_matrix_np=self.global_backrot_matrix,
                    )

                    # --- STAGE 1: PRE-CALIBRATION EVALUATION ---
                    self.console.print(
                        f"          Evaluating Pre-Calibration Zero-Shot Performance on all {len(epochs)} trials..."
                    )
                    pre_calib_preds = predictor.predict_batch(epochs, batch_size=self.args.batch_size_finetune)
                    pre_calib_metrics = score_predictions(
                        predictions=pre_calib_preds,
                        labels=labels_for_eval,
                        is_extreme_mask=is_extreme_mask,
                        original_soft_labels=labels_for_eval,
                    )
                    subject_results[model_name]["pre_calib_zero_shot"] = pre_calib_metrics
                    self.console.print(
                        f"          [bold]Pre-Calib ROC AUC: {pre_calib_metrics.get('roc_auc_all', np.nan):.4f}[/bold]"
                    )

                    # --- Split data for calibration and online phases ---
                    if metadata is not None and 'period' in metadata.columns:
                        cal_mask = (metadata['period'] == 'calibration').values
                        int_mask = (metadata['period'] == 'intervention').values
                        calibration_epochs = epochs[cal_mask]
                        online_epochs = epochs[int_mask]
                        calibration_labels_for_training = labels_ground_truth[cal_mask]
                        online_labels_for_finetuning = labels_ground_truth[int_mask]
                        online_labels_for_eval = labels_for_eval[int_mask]
                        online_is_extreme_mask = is_extreme_mask[int_mask]
                    else:
                        # No metadata: treat all data as online phase (no calibration)
                        calibration_epochs = None
                        online_epochs = epochs
                        online_labels_for_finetuning = labels_ground_truth
                        online_labels_for_eval = labels_for_eval
                        online_is_extreme_mask = is_extreme_mask

                    # --- STAGE 2: CALIBRATION ---
                    if calibration_epochs is not None and len(calibration_epochs) > 0:
                        self.console.print(
                            f"        Starting subject-specific calibration for {self.args.calibration_epochs} epochs..."
                        )
                        predictor.calibrate(calibration_epochs, calibration_labels_for_training)
                        self.console.print("        [green]Calibration complete.[/green]")

                    # --- STAGE 3: ONLINE EVALUATION ---
                    if online_epochs is not None and len(online_epochs) > 0:
                        self.console.print(
                            f"          Evaluating Post-Calibration Zero-Shot Performance on {len(online_epochs)} trials..."
                        )
                        post_calib_preds = predictor.predict_batch(
                            online_epochs, batch_size=self.args.batch_size_finetune
                        )
                        post_calib_metrics = score_predictions(
                            predictions=post_calib_preds,
                            labels=online_labels_for_eval,
                            is_extreme_mask=online_is_extreme_mask,
                            original_soft_labels=online_labels_for_eval,
                        )
                        subject_results[model_name]["post_calib_zero_shot"] = post_calib_metrics
                        self.console.print(
                            f"          [bold]Post-Calib ROC AUC: {post_calib_metrics.get('roc_auc_all', np.nan):.4f}[/bold]"
                        )

                        self.console.print(
                            f"        Starting online finetuning simulation on {len(online_epochs)} trials..."
                        )
                        final_finetuned_metrics, _, per_trial_metrics = run_online_finetuning_simulation(
                            predictor=predictor,
                            test_subj_epochs=online_epochs,
                            labels_for_finetuning=online_labels_for_finetuning,
                            labels_for_evaluation=online_labels_for_eval,
                            is_extreme_mask=online_is_extreme_mask,
                            original_soft_labels=online_labels_for_eval,
                            args=self.args, device=self.device, console=self.console, wandb_run=None,
                            run_output_dir=self.run_output_dir, model_name=model_name,
                            dataset_name=dataset_name,
                            subject_id=subject_id, fold_idx=fold_idx + 1,
                        )
                        subject_results[model_name]["finetuned"] = final_finetuned_metrics
                        all_models_trial_metrics[model_name] = per_trial_metrics

                        if self.args.get('save_finetuned_model', False):
                            save_path = self.run_output_dir / f"finetuned_subj_{subject_id}_fold_{fold_idx+1}.pt"
                            save_checkpoint(
                                {'model_state_dict': model_eval_wrapped.state_dict()},
                                save_path,
                            )
                            self.console.print(
                                f"        [green]Saved fine-tuned model to {save_path.name}[/green]"
                            )
                    else:
                        self.console.print(
                            "        No data available for the online phase. Final results will be empty."
                        )
                except Exception as e:
                    log.error(
                        f"Error processing model {model_name} for subject {subject_id}: {e}",
                        exc_info=True,
                    )
        except Exception as e:
            log.error(f"Critical error testing subject {subject_id}: {e}", exc_info=True)

        return subject_results, all_models_trial_metrics

    # ------------------------------------------------------------------
    # CONVENIENCE: FULL K-FOLD PIPELINE
    # ------------------------------------------------------------------

    def run_kfold(self, dataset_name: str) -> Tuple[dict, list]:
        """Execute the full k-fold cross-validation pipeline for a dataset.

        Handles subject splitting, data loading, training, and testing for
        each fold.

        Returns:
            Tuple of (results_per_model_per_fold, all_trial_metrics).
        """
        all_subjects = get_subject_list_for_datasets([dataset_name], self.args.data_root)
        subjects_to_run = (
            [s for s in all_subjects if s in self.args.subjects]
            if self.args.subjects else all_subjects
        )

        if len(subjects_to_run) < self.args.n_splits:
            log.warning(
                f"Insufficient subjects ({len(subjects_to_run)}) for {self.args.n_splits} splits. Skipping."
            )
            return {}, []

        kf = KFold(n_splits=self.args.n_splits, shuffle=True, random_state=self.args.seed)
        results_current_dataset = {m: {f: {} for f in range(self.args.n_splits)} for m in self.args.models_to_run}
        all_trial_metrics = []

        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(subjects_to_run)):
            train_subject_ids = [subjects_to_run[i] for i in train_indices]
            test_subject_ids = [subjects_to_run[i] for i in test_indices]

            run_only_fold = getattr(self.args, "run_only_fold", None)
            if run_only_fold is not None and (fold_idx + 1) != run_only_fold:
                continue

            # Reset RNG per-fold
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.args.seed)

            self.console.print(
                f"\n  [bold blue]=> Fold {fold_idx+1}/{self.args.n_splits} "
                f"| Train: {train_subject_ids} | Test: {test_subject_ids}[/bold blue]"
            )

            try:
                # --- TRAINING STAGE ---
                train_success = self._train_fold(dataset_name, fold_idx, train_subject_ids)
                if not train_success:
                    log.error(f"Could not prepare models for fold {fold_idx+1}. Skipping.")
                    continue

                # --- TESTING STAGE ---
                max_test_subjs = getattr(self.args, "max_test_subjects_per_fold", None)
                for subj_count, test_subject_id in enumerate(test_subject_ids):
                    test_epochs, test_labels, test_metadata = self._load_test_subject_data(
                        dataset_name, test_subject_id
                    )
                    subject_results, subject_trial_metrics = self.test(
                        epochs=test_epochs,
                        labels=test_labels,
                        metadata=test_metadata,
                        subject_id=test_subject_id,
                        fold_idx=fold_idx,
                        dataset_name=dataset_name,
                    )

                    for model_name, res_dict in subject_results.items():
                        results_current_dataset[model_name][fold_idx][test_subject_id] = res_dict
                    for model_name, trial_data in subject_trial_metrics.items():
                        for entry in trial_data:
                            entry.update({
                                "dataset": dataset_name, "model": model_name,
                                "fold": fold_idx + 1, "subject_id": test_subject_id
                            })
                            all_trial_metrics.append(entry)

                    if max_test_subjs is not None and (subj_count + 1) >= max_test_subjs:
                        self.console.print(
                            f"  [bold yellow]Reached max_test_subjects_per_fold={max_test_subjs}. Stopping early.[/bold yellow]"
                        )
                        break

            except Exception as e:
                log.error(f"Error processing Fold {fold_idx+1}: {e}", exc_info=True)

        return results_current_dataset, all_trial_metrics

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _train_fold(self, dataset_name: str, fold_idx: int, train_subject_ids: list) -> bool:
        """Load training data and run training for one fold."""
        if self.args.pretrained_checkpoint_dir:
            # Load from checkpoint
            log.info(f"Loading pre-trained models from: {self.args.pretrained_checkpoint_dir}")
            # Get data dimensions from a train subject
            _, _, n_ch, n_tp, _, _ = load_cached_pretrain_data(
                dataset_names=[dataset_name], subject_ids=[train_subject_ids[0]],
                paradigm_kwargs={
                    "fmin": self.args.fmin, "fmax": self.args.fmax,
                    "tmin": self.args.tmin, "tmax": self.args.tmax,
                    "resample": self.args.resample,
                },
                data_root=self.args.data_root, args=self.args
            )
            self.n_channels, self.n_timepoints = n_ch, n_tp
            return self.load_pretrained(Path(self.args.pretrained_checkpoint_dir), fold_idx)

        elif not self.args.no_pretrain:
            # Train from scratch: load data, then call train()
            train_epochs, train_labels = self._load_train_data(dataset_name, fold_idx, train_subject_ids)
            if train_epochs is None or train_epochs.size == 0:
                self.console.print(f"[yellow]No training data loaded for fold {fold_idx+1}. Skipping.[/yellow]")
                return False
            return self.train(train_epochs, train_labels, fold_idx=fold_idx, dataset_name=dataset_name)

        else:
            # No pretraining, just get dimensions
            log.info("`no_pretrain` is True. Models will be randomly initialized.")
            _, _, n_ch, n_tp, _, _ = load_cached_pretrain_data(
                dataset_names=[dataset_name], subject_ids=[train_subject_ids[0]],
                paradigm_kwargs={
                    "fmin": self.args.fmin, "fmax": self.args.fmax,
                    "tmin": self.args.tmin, "tmax": self.args.tmax,
                    "resample": self.args.resample,
                },
                data_root=self.args.data_root, args=self.args
            )
            self.n_channels, self.n_timepoints = n_ch, n_tp
            return n_ch > 0 and n_tp > 0

    def _load_train_data(self, dataset_name: str, fold_idx: int,
                         train_subject_ids: list) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load and return training data for the given subjects."""
        actual_pretrain_subject_ids = list(train_subject_ids)
        num_subjects_to_pretrain_on = getattr(self.args, "num_pretrain_subjects", "max")
        if isinstance(num_subjects_to_pretrain_on, int) and num_subjects_to_pretrain_on < len(train_subject_ids):
            if num_subjects_to_pretrain_on > 0:
                rng = np.random.RandomState(self.args.seed + fold_idx)
                actual_pretrain_subject_ids = rng.choice(
                    train_subject_ids, size=num_subjects_to_pretrain_on, replace=False
                ).tolist()
                self.console.print(
                    f"  Sub-sampling: Using {len(actual_pretrain_subject_ids)} subjects for pretraining"
                )
        self.console.print(
            f"  Loading pretraining data for {len(actual_pretrain_subject_ids)} subjects..."
        )

        paradigm_kwargs = {
            "fmin": self.args.fmin,
            "fmax": self.args.fmax,
            "tmin": self.args.tmin,
            "tmax": self.args.tmax,
            "resample": self.args.resample,
        }
        if hasattr(self.args, "channel_subset") and self.args.channel_subset:
            paradigm_kwargs["channels"] = self.args.channel_subset

        epochs_data, labels_data, n_ch, n_tp, _, global_backrot = load_cached_pretrain_data(
            dataset_names=[dataset_name],
            subject_ids=actual_pretrain_subject_ids,
            paradigm_kwargs=paradigm_kwargs,
            data_root=self.args.data_root,
            args=self.args,
            apply_trial_ablation=True,
        )

        if global_backrot is not None:
            self.global_backrot_matrix = global_backrot

        if epochs_data is None or epochs_data.size == 0:
            return None, None

        self.console.print(f"    Total pretrain trials: {len(epochs_data)}.")
        return epochs_data, labels_data

    def _load_test_subject_data(self, dataset_name: str,
                                test_subject_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame]]:
        """Load test data for a single subject."""
        test_metadata = None

        if dataset_name == "TMSEEGClassificationTEPfree":
            dataset = TMSEEGDatasetTEPfree(data_path=self.args.data_root)
            paradigm = TMSEEGClassificationTEPfree(tmin=self.args.tmin, tmax=self.args.tmax)
            test_epochs, test_labels, test_metadata = paradigm.get_data(
                dataset=dataset, subjects=[test_subject_id]
            )
        else:
            test_epochs, test_labels, _, _, _, _ = load_cached_pretrain_data(
                dataset_names=[dataset_name], subject_ids=[test_subject_id],
                paradigm_kwargs={
                    "fmin": self.args.fmin, "fmax": self.args.fmax,
                    "resample": self.args.resample,
                },
                data_root=self.args.data_root, args=self.args, apply_trial_ablation=False
            )

        return test_epochs, test_labels, test_metadata


# %%
# Memory and pretraining utilities

def log_memory_usage(stage: str, log_obj=None):
    """Log current memory usage for debugging."""
    if log_obj:
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        log_obj.debug(f"Memory usage at {stage}: {memory_mb:.1f} MB")

#%%
def run_online_finetuning_simulation(predictor, test_subj_epochs,
                                     labels_for_finetuning, labels_for_evaluation,
                                     is_extreme_mask, original_soft_labels,
                                     args, device, console, wandb_run,
                                     run_output_dir, model_name, dataset_name, subject_id, fold_idx):
    """
    Run an online finetuning simulation using an already-constructed OnlinePredictor.
    """
    if test_subj_epochs is None or test_subj_epochs.size == 0:
        log.error(f"Empty test data provided for Subj {subject_id}. Skipping simulation.")
        return {}, 0.0, []

    n_trials_subj = test_subj_epochs.shape[0]
    log_prefix = f"{dataset_name}_Fold_{fold_idx}_Subj_{subject_id}_{model_name}"
    log.info(f"Starting online simulation for {log_prefix} ({n_trials_subj} trials)")

    metrics_tracker = RegressionMetricsTracker(window_size=args.window_size)
    trial_times = []
    trial_metrics_log = []

    try:
        predictor.prepare_for_stream(args.seed)
        online_iterator = tqdm(range(n_trials_subj), desc=f"Online Sim ({log_prefix})", leave=False)
        for trial_idx in online_iterator:
            trial_start_time = time.time()
            try:
                single_epoch_np = test_subj_epochs[trial_idx]
                single_label_for_finetuning = labels_for_finetuning[trial_idx]
                single_label_for_evaluation = labels_for_evaluation[trial_idx]

                # --- PREDICT ---
                pred_prob = predictor.predict(single_epoch_np)

                # --- FINETUNE (adapt alignment + buffered supervised update) ---
                step_loss_or_none = predictor.finetune(single_epoch_np, single_label_for_finetuning)
                step_loss = step_loss_or_none if step_loss_or_none is not None else np.nan

                metrics_tracker.update(
                    y_true=single_label_for_evaluation,
                    y_pred=pred_prob,
                )

                trial_times.append(time.time() - trial_start_time)

                online_iterator.set_postfix(
                    auc=f"{metrics_tracker.get_rolling_roc_auc():.3f}",
                )

                trial_metrics_log.append({
                    'trial_idx': trial_idx,
                    'rolling_roc_auc': metrics_tracker.get_rolling_roc_auc(),
                    'overall_roc_auc_at_trial': metrics_tracker.get_overall_roc_auc(),
                    'finetune_loss': step_loss,
                })
                if wandb_run:
                    wandb_run.log({f"online/{log_prefix}/rolling_roc_auc": metrics_tracker.get_rolling_roc_auc()}, step=trial_idx)
            except Exception as e:
                log.error(f"Error processing trial {trial_idx} for {log_prefix}: {e}", exc_info=True)
                continue

        # --- Final Metrics Calculation ---
        avg_time_per_trial = np.mean(trial_times) if trial_times else 0.0
        y_true_all = np.array(metrics_tracker.all_y_true)
        y_pred_all = np.array(metrics_tracker.all_y_pred)

        final_metrics = score_predictions(
            predictions=y_pred_all,
            labels=y_true_all,
            is_extreme_mask=is_extreme_mask,
            original_soft_labels=original_soft_labels,
        )

        log.info(f"Online sim finished. ROC AUC (All): {final_metrics.get('roc_auc_all', np.nan):.4f}.")

        if args.get('save_predictions_and_labels', False):
            output_filename = run_output_dir / f"predictions_subj_{subject_id}_fold_{fold_idx}.npz"
            np.savez_compressed(output_filename, predictions=y_pred_all, actual_values=y_true_all)
            log.info(f"Saved predictions and actuals for Subj {subject_id} to {output_filename}")

        if wandb_run: wandb_run.log({f"final/{log_prefix}": final_metrics})

        return final_metrics, avg_time_per_trial, trial_metrics_log

    except Exception as e:
        log.error(f"Critical error in online simulation for {log_prefix}: {e}", exc_info=True)
        return {}, 0.0, []
    finally:
        del metrics_tracker
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

# %% --- Experiment Execution and Result Aggregation ---

def run_cross_subject_experiment(
    args: OmegaConf, device: torch.device,
    console: Console, run_output_dir: Path
) -> Tuple[dict, list, list]:
    """Manages the cross-subject k-fold cross-validation experiment using CrossValidator."""
    console.print("\n[bold magenta]===== Starting Cross-Subject K-Fold Experiment =====[/bold magenta]")
    all_datasets_results = {}
    processed_dataset_names = []
    all_trial_metrics = []

    for dataset_name in args.dataset_names:
        console.print(f"\n[bold cyan]### Dataset: {dataset_name} ###[/bold cyan]")

        cv = CrossValidator(args, device, console, run_output_dir)
        results_current_dataset, dataset_trial_metrics = cv.run_kfold(dataset_name)

        if results_current_dataset:
            processed_dataset_names.append(dataset_name)
            all_datasets_results[dataset_name] = results_current_dataset
            all_trial_metrics.extend(dataset_trial_metrics)

    return all_datasets_results, processed_dataset_names, all_trial_metrics


def run_single_subject_eval(
    args: OmegaConf, device: torch.device,
    console: Console, run_output_dir: Path
) -> Tuple[dict, list, list]:
    """Evaluate a pretrained model on locally available subjects without cross-validation."""
    console.print("\n[bold magenta]===== Starting Single-Subject Evaluation =====[/bold magenta]")

    if not args.pretrained_checkpoint_dir:
        raise ValueError("single_subject_eval mode requires 'pretrained_checkpoint_dir' to be set.")

    all_datasets_results = {}
    processed_dataset_names = []
    all_trial_metrics = []

    for dataset_name in args.dataset_names:
        console.print(f"\n[bold cyan]### Dataset: {dataset_name} ###[/bold cyan]")

        all_subjects = get_subject_list_for_datasets([dataset_name], args.data_root)
        subjects_to_run = [s for s in all_subjects if s in args.subjects] if args.subjects else all_subjects

        if not subjects_to_run:
            log.warning(f"No subjects available for dataset '{dataset_name}'. Skipping.")
            continue

        processed_dataset_names.append(dataset_name)
        results_current_dataset = {m: {0: {}} for m in args.models_to_run}

        fold_idx = (getattr(args, "run_only_fold", 1) or 1) - 1

        # Seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        # Create CrossValidator and load pretrained weights
        cv = CrossValidator(args, device, console, run_output_dir)

        # Get data dimensions from a sample subject
        _, _, n_ch, n_tp, _, _ = load_cached_pretrain_data(
            dataset_names=[dataset_name], subject_ids=[subjects_to_run[0]],
            paradigm_kwargs={"fmin": args.fmin, "fmax": args.fmax, "tmin": args.tmin, "tmax": args.tmax, "resample": args.resample},
            data_root=args.data_root, args=args
        )
        cv.n_channels, cv.n_timepoints = n_ch, n_tp

        if not cv.load_pretrained(Path(args.pretrained_checkpoint_dir), fold_idx):
            log.error("No pretrained models could be loaded. Skipping dataset.")
            continue

        # --- Evaluate each local subject ---
        max_test_subjs = getattr(args, "max_test_subjects_per_fold", None)
        console.print(f"  Evaluating on {len(subjects_to_run)} subject(s): {subjects_to_run}")

        for subj_count, test_subject_id in enumerate(subjects_to_run):
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

            test_epochs, test_labels, test_metadata = cv._load_test_subject_data(
                dataset_name, test_subject_id
            )
            subject_results, subject_trial_metrics = cv.test(
                epochs=test_epochs,
                labels=test_labels,
                metadata=test_metadata,
                subject_id=test_subject_id,
                fold_idx=fold_idx,
                dataset_name=dataset_name,
            )

            for model_name, res_dict in subject_results.items():
                results_current_dataset[model_name][0][test_subject_id] = res_dict
            for model_name, trial_data in subject_trial_metrics.items():
                for entry in trial_data:
                    entry.update({"dataset": dataset_name, "model": model_name, "fold": 1, "subject_id": test_subject_id})
                    all_trial_metrics.append(entry)

            if max_test_subjs is not None and (subj_count + 1) >= max_test_subjs:
                console.print(f"  [bold yellow]Reached max_test_subjects_per_fold={max_test_subjs}. Stopping.[/bold yellow]")
                break

        all_datasets_results[dataset_name] = results_current_dataset

    return all_datasets_results, processed_dataset_names, all_trial_metrics


def aggregate_and_report_results(results: dict, dataset_names: list, args: OmegaConf,
                                 console: Console, run_output_dir: Path):
    """Aggregates, averages, and reports all experimental results.
    """
    console.print("\n[bold magenta]===== Aggregated Results Summary =====[/bold magenta]")
    
    # Define the evaluation stages and metrics to be reported
    STAGES = {
        "pre_calib_zero_shot": "Pre-Calibration",
        "post_calib_zero_shot": "Post-Calibration",
        "finetuned": "Online Fine-tuned"
    }
    METRICS = {
        "roc_auc_all": "ROC AUC (All)",
        "roc_auc_extreme": "ROC AUC (Extreme)",
    }

    # --- Data Aggregation ---
    # Average results across folds for each subject
    final_subject_metrics = {}
    for ds_name, ds_results in results.items():
        final_subject_metrics[ds_name] = {m: {} for m in ds_results}
        for model_name, model_results in ds_results.items():
            # Collect all results for each subject across all folds
            subj_fold_data = {}
            for fold_data in model_results.values():
                for subj_id, metric_dict in fold_data.items():
                    if subj_id not in subj_fold_data:
                        subj_fold_data[subj_id] = {stage: [] for stage in STAGES}
                    for stage_key, stage_metrics in metric_dict.items():
                        subj_fold_data[subj_id][stage_key].append(stage_metrics)
            
            # Average the metrics
            for subj_id, collected_metrics in subj_fold_data.items():
                final_subject_metrics[ds_name][model_name][subj_id] = {}
                for stage_key, stage_list in collected_metrics.items():
                    df = pd.DataFrame(stage_list)
                    final_subject_metrics[ds_name][model_name][subj_id][stage_key] = df.mean().to_dict()

    # --- Reporting ---
    csv_rows = []
    for ds_name in dataset_names:
        console.print(f"\n[bold green]### Results for Dataset: {ds_name} ###[/bold green]")
        ds_metrics = final_subject_metrics.get(ds_name, {})
        all_subjects = sorted({sid for res in ds_metrics.values() for sid in res})

        for stage_key, stage_title in STAGES.items():
            console.print(f"\n[bold]--- {stage_title} Performance ---[/bold]")
            for metric_key, metric_title in METRICS.items():
                # Check if there is any data for this metric to avoid empty tables
                has_data = any(metric_key in (ds_metrics.get(m, {}).get(s, {}).get(stage_key, {})) for m in ds_metrics for s in all_subjects)
                if not has_data: continue

                table = Table(title=f"{metric_title} - {ds_name}")
                table.add_column("Model", style="cyan")
                for sid in all_subjects: table.add_column(f"Subj {sid}", justify="right")
                table.add_column("Mean", style="green", justify="right")

                for model_name, model_data in ds_metrics.items():
                    scores = [model_data.get(sid, {}).get(stage_key, {}).get(metric_key, np.nan) for sid in all_subjects]
                    row = [model_name] + [f"{s:.4f}" if not np.isnan(s) else "N/A" for s in scores]
                    row.append(f"[bold]{np.nanmean(scores):.4f}[/bold]")
                    table.add_row(*row)
                console.print(table)
        
        # Collect data for CSV export
        for model_name, model_data in ds_metrics.items():
            for subject_id, subject_data in model_data.items():
                row = {"dataset": ds_name, "model": model_name, "subject_id": subject_id}
                for stage_key, stage_metrics in subject_data.items():
                    for metric_key, val in stage_metrics.items():
                        row[f"{stage_key}_{metric_key}"] = val
                csv_rows.append(row)

    # --- Save to CSV ---
    if args.save_results and csv_rows:
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(run_output_dir / "results_summary.csv", index=False)
        log.info(f"Aggregated results summary saved to {run_output_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    # 1. Initialize experiment: parse args, create directories, set seed
    log = logging.getLogger(__name__)
    try:
        args, device, console, run_output_dir = setup_experiment()
        # Setup file logging to save run details
        file_handler = logging.FileHandler(run_output_dir / "run.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
        log.info("Final Configuration:\n%s", OmegaConf.to_yaml(args))
    except Exception as e:
        log.error(f"Experiment setup failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Run the main experiment logic
    try:
        if args.experiment_mode == "cross_subject_kfold":
            all_results, processed_datasets, trial_metrics = run_cross_subject_experiment(
                args, device, console, run_output_dir
            )
        elif args.experiment_mode == "single_subject_eval":
            all_results, processed_datasets, trial_metrics = run_single_subject_eval(
                args, device, console, run_output_dir
            )
        else:
            raise ValueError(f"Unknown experiment_mode: '{args.experiment_mode}'")
        
        # 3. Aggregate and report results
        if not all_results:
            log.warning("Experiment finished but no results were generated.")
        else:
            aggregate_and_report_results(all_results, processed_datasets, args, console, run_output_dir)
        
        # Save trial-by-trial metrics
        if args.save_results and trial_metrics:
            trial_df = pd.DataFrame(trial_metrics)
            trial_df.to_csv(run_output_dir / "results_trial_metrics.csv", index=False)
            log.info(f"Trial-by-trial metrics saved to {run_output_dir / 'results_trial_metrics.csv'}")
            
        console.print("\n[bold green]✅ Experiment Complete.[/bold green]")
        log.info("Experiment finished successfully.")

    except KeyboardInterrupt:
        log.warning("Experiment interrupted by user (Ctrl+C).")
    except Exception as e:
        log.critical(f"A critical error occurred during the main execution: {e}", exc_info=True)
    finally:
        # 4. Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory_usage("final_cleanup", log)
        console.print("\n[dim]Cleanup complete.[/dim]")

