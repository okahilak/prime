#%%
"""
Main script for conducting transfer learning experiments on EEG-TMS data using
cross-subject k-fold cross-validation.

This script handles:
- Pre-training the PRIME model on a set of source subjects.
- Evaluating the model in a zero-shot manner on a target subject.
- Performing subject-specific calibration.
- Simulating online fine-tuning on a trial-by-trial basis.
- Aggregating and reporting performance metrics.
"""

# Core Python libraries
import argparse
import gc
import logging
import os
import sys
import warnings
from pathlib import Path

# Third-party libraries for data handling and computation
import mne
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from omegaconf import ListConfig
from rich.table import Table

# Visualization and utility libraries
from rich.console import Console

# Local project-specific modules
from datasets import *
from cross_validator import CrossValidator, log_memory_usage
from utils import get_output_dir, save_results_df

# --- Constants ---
DATASET_NAME = "TEP"
MODEL_NAME = "PRIME"

# --- Global Setup ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# %%
# Experiment setup

def setup_experiment(cli_args=None):
    """Parse arguments, setup configuration, and initialize experiment."""

    DEFAULT_YAML = """
subjects: null
data_root: "~/prime-data/processed"
pretrained_checkpoint_dir: null
fmin: null
fmax: null
tmin: -0.55
tmax: -0.050
resample: null
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
optimizer_type_finetune: "AdamW"
weight_decay_finetune: 0.0
batch_size_finetune: 50
seed: 42
device: "cuda"
no_pretrain: false
base_output_dir: "results"
save_pretrained_model: false
save_finetuned_model: false
save_checkpoints: false
save_results: true
save_predictions_and_labels: true
num_pretrain_subjects: "max"
num_trials_per_subject: null

# TTA Configuration
use_tta: true
ea_backrotation: true
alignment_type: "euclidean"
alignment_cov_epsilon: 1.0e-6
alignment_transform_epsilon: 1.0e-7
alignment_ref_ema_beta: 0.99
tta_cov_buffer_size: 50
use_adabn: false
finetune_mode: "full"

experiment_mode: "cross_subject_kfold"

use_subject_specific_calibration: true
num_calibration_trials: 100
lr_calibration: 0.0001
calibration_epochs: 50

shuffle_test_labels: false
max_test_subjects_per_fold: null
"""

    config = OmegaConf.create(DEFAULT_YAML)

    warnings.filterwarnings("ignore", category=UserWarning, module="moabb")
    warnings.filterwarnings("ignore", message="warnEpochs*")
    logging.getLogger("moabb").setLevel(logging.ERROR)
    logging.getLogger("moabb.paradigms").setLevel(logging.ERROR)
    logging.getLogger("moabb.datasets").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="K-Fold Transfer with YAML Config")
    parser.add_argument("-c", "--config", action="append", help="Path to YAML config file(s)", default=[])

    parsed_args, remaining_argv = parser.parse_known_args(args=cli_args)

    if parsed_args.config:
        for config_file in parsed_args.config:
            user_config = OmegaConf.load(config_file)
            config = OmegaConf.merge(config, user_config)
            print(f"Loaded config from: {config_file}")

    if remaining_argv:
        cli_conf = OmegaConf.from_cli(remaining_argv)
        if cli_conf:
            config = OmegaConf.merge(config, cli_conf)

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

    # Create output directory
    config_name = ""
    if parsed_args.config:
        config_name = Path(parsed_args.config[-1]).stem
    run_output_dir = get_output_dir(
        base_output_root=config.base_output_dir,
        config_name=config_name,
    )
    console = Console()
    console.print(f"[blue]Output directory: {run_output_dir}[/blue]")

    config_save_path = run_output_dir / "config.yaml"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_save_path)

    device = torch.device(config.device)
    if config.device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available. Switching to CPU.[/yellow]")
        device = torch.device("cpu")
        config.device = "cpu"

    # Process subjects
    if config.subjects is not None:
        if isinstance(config.subjects, int):
            config.subjects = [config.subjects]
        elif isinstance(config.subjects, (list, ListConfig)):
            config.subjects = [int(s) for s in config.subjects]

    return config, device, console, run_output_dir


# %% --- Experiment Execution and Result Aggregation ---

def run_cross_subject_experiment(args, device, console, run_output_dir):
    """Run the cross-subject k-fold experiment."""
    console.print("\n[bold magenta]===== Starting Cross-Subject K-Fold Experiment =====[/bold magenta]")
    cv = CrossValidator(args, device, console, run_output_dir)
    fold_results, trial_metrics = cv.run_kfold()
    return fold_results, trial_metrics


def run_train_only(args, device, console, run_output_dir):
    """Train a classifier on all (or selected) subjects and save it. No CV or testing."""
    console.print("\n[bold magenta]===== Train-Only Mode =====[/bold magenta]")

    all_subjects = get_subject_list(args.data_root)
    subjects_to_run = (
        [s for s in all_subjects if s in args.subjects]
        if args.subjects else all_subjects
    )
    assert subjects_to_run, "No subjects available for training."

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Always save the pretrained model in this mode
    args.save_pretrained_model = True

    cv = CrossValidator(args, device, console, run_output_dir)
    train_epochs, train_labels = cv._load_train_data(fold_idx=0, train_subject_ids=subjects_to_run)
    assert train_epochs is not None and train_epochs.size > 0, "No training data loaded."
    cv.train(train_epochs, train_labels, fold_idx=0)

    console.print("[bold green]Training complete. Model saved to output directory.[/bold green]")


def run_single_subject_eval(args, device, console, run_output_dir):
    """Evaluate a pretrained model on locally available subjects."""
    console.print("\n[bold magenta]===== Starting Single-Subject Evaluation =====[/bold magenta]")
    assert args.pretrained_checkpoint_dir, \
        "single_subject_eval mode requires 'pretrained_checkpoint_dir' to be set."

    all_subjects = get_subject_list(args.data_root)
    subjects_to_run = [s for s in all_subjects if s in args.subjects] if args.subjects else all_subjects
    assert subjects_to_run, "No subjects available."

    fold_idx = (getattr(args, "run_only_fold", 1) or 1) - 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cv = CrossValidator(args, device, console, run_output_dir)

    # Get data dimensions
    _, _, n_ch, n_tp, _, _ = load_pretrain_data(
        subject_ids=[subjects_to_run[0]],
        paradigm_kwargs={"fmin": args.fmin, "fmax": args.fmax, "tmin": args.tmin,
                         "tmax": args.tmax, "resample": args.resample},
        data_root=args.data_root, args=args,
    )
    cv.n_channels, cv.n_timepoints = n_ch, n_tp
    cv.load_pretrained(Path(args.pretrained_checkpoint_dir), fold_idx)

    fold_results = {0: {}}
    all_trial_metrics = []
    max_test_subjs = getattr(args, "max_test_subjects_per_fold", None)
    console.print(f"  Evaluating on {len(subjects_to_run)} subject(s): {subjects_to_run}")

    for subj_count, test_subject_id in enumerate(subjects_to_run):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        test_epochs, test_labels, test_metadata = cv._load_test_subject_data(test_subject_id)
        subject_results, subject_trial_metrics = cv.test(
            epochs=test_epochs, labels=test_labels,
            metadata=test_metadata,
            subject_id=test_subject_id, fold_idx=fold_idx,
        )

        fold_results[0][test_subject_id] = subject_results
        for entry in subject_trial_metrics:
            entry.update({"fold": 1, "subject_id": test_subject_id})
            all_trial_metrics.append(entry)

        if max_test_subjs is not None and (subj_count + 1) >= max_test_subjs:
            console.print(f"  [bold yellow]Reached max_test_subjects_per_fold={max_test_subjs}. Stopping.[/bold yellow]")
            break

    return fold_results, all_trial_metrics


def aggregate_and_report_results(fold_results: dict, args: OmegaConf,
                                 console: Console, run_output_dir: Path):
    """Aggregates, averages, and reports experimental results."""
    console.print("\n[bold magenta]===== Aggregated Results Summary =====[/bold magenta]")

    STAGES = {
        "pre_calib_zero_shot": "Pre-Calibration",
        "post_calib_zero_shot": "Post-Calibration",
        "finetuned": "Online Fine-tuned",
    }
    METRICS = {
        "roc_auc_all": "ROC AUC (All)",
        "roc_auc_extreme": "ROC AUC (Extreme)",
    }

    # Aggregate across folds per subject
    subj_fold_data = {}
    for fold_data in fold_results.values():
        for subj_id, metric_dict in fold_data.items():
            if subj_id not in subj_fold_data:
                subj_fold_data[subj_id] = {stage: [] for stage in STAGES}
            for stage_key, stage_metrics in metric_dict.items():
                subj_fold_data[subj_id][stage_key].append(stage_metrics)

    # Average
    final_metrics = {}
    for subj_id, collected_metrics in subj_fold_data.items():
        final_metrics[subj_id] = {}
        for stage_key, stage_list in collected_metrics.items():
            df = pd.DataFrame(stage_list)
            final_metrics[subj_id][stage_key] = df.mean().to_dict()

    all_subjects = sorted(final_metrics.keys())

    for stage_key, stage_title in STAGES.items():
        console.print(f"\n[bold]--- {stage_title} Performance ---[/bold]")
        for metric_key, metric_title in METRICS.items():
            has_data = any(metric_key in final_metrics.get(s, {}).get(stage_key, {}) for s in all_subjects)
            if not has_data:
                continue

            table = Table(title=f"{metric_title}")
            table.add_column("Model", style="cyan")
            for sid in all_subjects:
                table.add_column(f"Subj {sid}", justify="right")
            table.add_column("Mean", style="green", justify="right")

            scores = [final_metrics.get(sid, {}).get(stage_key, {}).get(metric_key, np.nan) for sid in all_subjects]
            row = [MODEL_NAME] + [f"{s:.4f}" if not np.isnan(s) else "N/A" for s in scores]
            row.append(f"[bold]{np.nanmean(scores):.4f}[/bold]")
            table.add_row(*row)
            console.print(table)

    # CSV export
    csv_rows = []
    for subject_id, subject_data in final_metrics.items():
        row_data = {"model": MODEL_NAME, "subject_id": subject_id}
        for stage_key, stage_metrics in subject_data.items():
            for metric_key, val in stage_metrics.items():
                row_data[f"{stage_key}_{metric_key}"] = val
        csv_rows.append(row_data)

    if args.save_results and csv_rows:
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(run_output_dir / "results_summary.csv", index=False)
        log.info(f"Results saved to {run_output_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    args, device, console, run_output_dir = setup_experiment()

    file_handler = logging.FileHandler(run_output_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    log.info("Final Configuration:\n%s", OmegaConf.to_yaml(args))

    if args.experiment_mode == "train_only":
        run_train_only(args, device, console, run_output_dir)
        console.print("\n[bold green]\u2705 Training Complete.[/bold green]")
        log.info("Training finished successfully.")
        sys.exit(0)
    elif args.experiment_mode == "cross_subject_kfold":
        fold_results, trial_metrics = run_cross_subject_experiment(args, device, console, run_output_dir)
    elif args.experiment_mode == "single_subject_eval":
        fold_results, trial_metrics = run_single_subject_eval(args, device, console, run_output_dir)
    else:
        raise ValueError(f"Unknown experiment_mode: '{args.experiment_mode}'")

    if fold_results:
        aggregate_and_report_results(fold_results, args, console, run_output_dir)

    if args.save_results and trial_metrics:
        trial_df = pd.DataFrame(trial_metrics)
        trial_df.to_csv(run_output_dir / "results_trial_metrics.csv", index=False)
        log.info(f"Trial metrics saved to {run_output_dir / 'results_trial_metrics.csv'}")

    console.print("\n[bold green]✅ Experiment Complete.[/bold green]")
    log.info("Experiment finished successfully.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory_usage("final_cleanup", log)
    console.print("\n[dim]Cleanup complete.[/dim]")
