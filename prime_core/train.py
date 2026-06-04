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

# Force single-threaded BLAS/LAPACK BEFORE importing numpy/scipy/torch.
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
from rich.table import Table

# Visualization and utility libraries
from rich.console import Console

# Local project-specific modules
from prime_core.datasets import *
from prime_core.cross_validator import CrossValidator, log_memory_usage
from prime_core.prime_config import PRIME_CONFIG_PATH
from prime_core.utils import save_results_df

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

    warnings.filterwarnings("ignore", category=UserWarning, module="moabb")
    warnings.filterwarnings("ignore", message="warnEpochs*")
    logging.getLogger("moabb").setLevel(logging.ERROR)
    logging.getLogger("moabb.paradigms").setLevel(logging.ERROR)
    logging.getLogger("moabb.datasets").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="PRIME transfer learning")

    # Mode (mutually exclusive, required)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train-all", action="store_true",
                            help="Train on all subjects, save as pretrained_all.pt")
    mode_group.add_argument("--train", type=int, nargs="+", metavar="SUBJECT",
                            help="Train on specified subjects (e.g. --train 1 2 3 4)")
    mode_group.add_argument("--test", type=int, nargs="+", metavar="SUBJECT",
                            help="Test on specified subjects (e.g. --test 1 2 3)")
    mode_group.add_argument("--cv", action="store_true",
                            help="Full cross-subject k-fold cross-validation")

    parser.add_argument("-c", "--config", action="append", default=None,
                        help=f"Path to YAML config file(s) (default: {PRIME_CONFIG_PATH})")
    parser.add_argument("--n-splits", type=int, default=None,
                        help="Number of CV splits (default: 2)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold (1-based; for --cv)")

    parsed_args, remaining_argv = parser.parse_known_args(args=cli_args)

    if parsed_args.config is None:
        parsed_args.config = [str(PRIME_CONFIG_PATH)]

    config = OmegaConf.load(parsed_args.config[0])
    print(f"Loaded config from: {parsed_args.config[0]}")
    for config_file in parsed_args.config[1:]:
        user_config = OmegaConf.load(config_file)
        config = OmegaConf.merge(config, user_config)
        print(f"Loaded config from: {config_file}")

    if remaining_argv:
        cli_conf = OmegaConf.from_cli(remaining_argv)
        if cli_conf:
            config = OmegaConf.merge(config, cli_conf)

    OmegaConf.resolve(config)

    # Apply CLI overrides for mode, subjects, n_splits
    if parsed_args.train_all:
        config.experiment_mode = "train_only"
        config.subjects = None  # all subjects
        config.train_all = True
    elif parsed_args.train:
        config.experiment_mode = "train_only"
        config.subjects = parsed_args.train
        config.train_all = False
    elif parsed_args.test:
        config.experiment_mode = "single_subject_eval"
        config.pretrained_checkpoint_dir = str(Path(config.base_output_dir) / "train")
        config.subjects = parsed_args.test
        config.train_all = False
    elif parsed_args.cv:
        config.experiment_mode = "cross_subject_kfold"
        config.subjects = None  # all subjects for CV
        config.train_all = False

    config.n_splits = parsed_args.n_splits if parsed_args.n_splits is not None else 2
    config.num_pretrain_subjects = "max"

    if parsed_args.fold is not None:
        config.run_only_fold = parsed_args.fold

    config.data_root = str(Path(config.data_root).expanduser())

    # Deterministic CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Create output directory
    if parsed_args.train_all or parsed_args.train:
        run_output_dir = Path(config.base_output_dir) / "train"
    elif parsed_args.test:
        run_output_dir = Path(config.base_output_dir) / "test"
    else:
        run_output_dir = Path(config.base_output_dir) / "cv"
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

    return config, device, console, run_output_dir


# %% --- Experiment Execution and Result Aggregation ---

def run_cross_subject_experiment(args, device, console, run_output_dir):
    """Run the cross-subject k-fold experiment."""
    console.print("\n[bold magenta]===== Starting Cross-Subject K-Fold Experiment =====[/bold magenta]")
    cv = CrossValidator(args, device, console, run_output_dir, is_cv=True)
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

    cv = CrossValidator(args, device, console, run_output_dir)
    train_epochs, train_labels = cv._load_train_data(train_subject_ids=subjects_to_run)
    assert train_epochs is not None and train_epochs.size > 0, "No training data loaded."
    cv.train(train_epochs, train_labels)

    console.print("[bold green]Training complete. Model saved to output directory.[/bold green]")


def run_test_only(args, device, console, run_output_dir):
    """Evaluate a pretrained model on locally available subjects."""
    console.print("\n[bold magenta]===== Starting Test-Only Evaluation =====[/bold magenta]")
    assert args.pretrained_checkpoint_dir, \
        "--test mode requires a pretrained model in results/train/."

    all_subjects = get_subject_list(args.data_root)
    subjects_to_run = [s for s in all_subjects if s in args.subjects] if args.subjects else all_subjects
    assert subjects_to_run, "No subjects available."

    cv = CrossValidator(args, device, console, run_output_dir)

    # Get data dimensions
    _, _, n_ch, n_tp, _, _ = load_pretrain_data(
        subject_ids=[subjects_to_run[0]],
        paradigm_kwargs={"fmin": args.fmin, "fmax": args.fmax, "resample": args.resample},
        data_root=args.data_root, args=args,
    )
    cv.n_channels, cv.n_timepoints = n_ch, n_tp
    cv.load_pretrained(Path(args.pretrained_checkpoint_dir))

    fold_results = {0: {}}
    all_trial_metrics = []
    console.print(f"  Evaluating on {len(subjects_to_run)} subject(s): {subjects_to_run}")

    for test_subject_id in subjects_to_run:
        test_epochs, test_labels, test_metadata = cv._load_test_subject_data(test_subject_id)
        subject_results, subject_trial_metrics = cv.test(
            epochs=test_epochs, labels=test_labels,
            metadata=test_metadata,
            subject_id=test_subject_id,
        )

        fold_results[0][test_subject_id] = subject_results
        for entry in subject_trial_metrics:
            entry.update({"subject_id": test_subject_id})
            all_trial_metrics.append(entry)

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
            row = ["PRIME"] + [f"{s:.4f}" if not np.isnan(s) else "N/A" for s in scores]
            row.append(f"[bold]{np.nanmean(scores):.4f}[/bold]")
            table.add_row(*row)
            console.print(table)

    # CSV export
    csv_rows = []
    for subject_id, subject_data in final_metrics.items():
        row_data = {"model": "PRIME", "subject_id": subject_id}
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
        fold_results, trial_metrics = run_test_only(args, device, console, run_output_dir)

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
