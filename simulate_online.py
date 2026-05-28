"""
Simulate real-time (online) processing for a single subject, trial by trial.

This script processes data sequentially, one trial at a time, to mirror
what a true online system would do. Currently implements the calibration phase:

1. Feed trials one at a time, accumulating pre/post/ica epochs.
2. Once N_CALIBRATION_TRIALS are accumulated, run batch calibration preprocessing.
3. After calibration: calibrate dipole parameters from the averaged post-stim data.
4. Preprocess each calibration trial's pre and post periods using calibration params.
5. Fit dipoles to the calibration post-stim epochs.

Hard-coded settings:
- Subject: sub-021
- Number of calibration trials: 125
"""

import copy
import sys
import time
import warnings
from collections import deque
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")

# --- Local imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "online_preprocessing"))

from online_preprocessing.config import get_default_config
from online_preprocessing.calibrator import Calibrator
from online_preprocessing.dipole_fitter import DipoleFitter
from online_preprocessing.preprocess import (
    _load_subject_epochs,
    _single_trial_epochs_from_arrays,
)
from TMS_EEG_moabb import TEPNormalizer
from models.builder import build_model
from tta_wrapper import TTAWrapper, _apply_alignment_transform_np

# =============================================================================
# Hard-coded constants
# =============================================================================
SUBJECT_ID = "sub-021"
N_CALIBRATION_TRIALS = 125
DATA_ROOT = Path("~/prime-data").expanduser()

# Offline results directory (for pretrained model & comparison)
PREDICTIONS_PATH = (
    "results/2026-05-28_16-41-27_eval_single_subject/predictions_subj_21_fold_1.npz"
)

PRETRAINED_MODEL_PATH = "pretrained_fold_1.pt"
GLOBAL_BACKROT_PATH = "global_backrotation_matrix_fold_1.npy"

# Config matching replicate_prime.yaml
CONFIG = {
    "tmin": -0.060,
    "tmax": -0.010,
    "use_tta": True,
    "alignment_type": "euclidean",
    "ea_backrotation": True,
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
}


def main():
    print("=" * 70)
    print("SIMULATE ONLINE PROCESSING (trial-by-trial)")
    print(f"Subject: {SUBJECT_ID}")
    print(f"Calibration trials: {N_CALIBRATION_TRIALS}")
    print("=" * 70)

    # --- Setup ---
    cfg = get_default_config()
    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"

    # --- Load all data (in a real system, trials would arrive one at a time) ---
    print("\nLoading raw data...")
    epochs = _load_subject_epochs(SUBJECT_ID, cfg)
    all_eeg_data = epochs.get_data(copy=False)
    all_events = epochs.events
    n_total_trials = all_eeg_data.shape[0]
    print(f"Loaded {n_total_trials} total trials for {SUBJECT_ID}")

    # =========================================================================
    # CALIBRATION PHASE — feed trials one at a time
    # =========================================================================
    print("\n" + "=" * 70)
    print("CALIBRATION PHASE")
    print("=" * 70)

    # --- Trial-by-trial accumulation ---
    calibrator = Calibrator(cfg, forward_path)

    for trial_idx in range(N_CALIBRATION_TRIALS):
        # Simulate receiving a single trial
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)

        calibrator.add_trial(trial)

        if (trial_idx + 1) % 25 == 0:
            print(f"  Accumulated {trial_idx + 1}/{N_CALIBRATION_TRIALS} calibration trials")

    # --- We have enough trials: run calibration ---
    print("\nRunning calibration preprocessing...")
    t0 = time.time()
    n_successful_trials = calibrator.calibrate()
    print(f"Calibration preprocessing done in {time.time() - t0:.2f}s, "
          f"used {n_successful_trials} trials")

    # --- Process calibration trials: pre + post together (offline requires both) ---
    print("\nPreprocessing calibration trials (both pre & post)...")
    cal_pre_list = []
    cal_post_list = []
    for trial_idx in range(N_CALIBRATION_TRIALS):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        result_pre = calibrator.preprocess_pre(trial)
        result_post = calibrator.preprocess_post(trial)
        if result_pre is not False and result_post is not False:
            cal_pre_list.append(result_pre)
            cal_post_list.append(result_post)

    print(f"  {len(cal_post_list)}/{N_CALIBRATION_TRIALS} calibration trials survived "
          f"(both pre & post)")
    cal_pre_epochs = mne.concatenate_epochs(cal_pre_list)
    cal_post_epochs = mne.concatenate_epochs(cal_post_list)

    # --- Calibrate dipole fitting parameters ---
    print("\nCalibrating dipole parameters...")
    dipole_fitter = DipoleFitter(forward_path)
    dipole_fitter.fit(cal_post_epochs)
    position_index = dipole_fitter.fitting_info['position_index']
    tmin_fit, tmax_fit = dipole_fitter.fitting_info['time_range']
    print(f"  Position index: {position_index}")
    print(f"  Time range: [{tmin_fit*1000:.1f}, {tmax_fit*1000:.1f}] ms")

    # --- Fit dipoles to calibration post-stim trials ---
    print("\nFitting dipoles to calibration trials...")
    cal_dipoles_free = None
    for orientation_label, orientation in [('fixed', 'use_fitted'), ('free', None)]:
        dipoles_for_trials, extraction_times = dipole_fitter.fit_trials(
            cal_post_epochs, orientation=orientation)
        mean_time_ms = np.mean(extraction_times) * 1e3
        print(f"  {orientation_label} orientation: {len(dipoles_for_trials)} dipoles, "
              f"avg extraction time {mean_time_ms:.2f} ms")
        if orientation is None:
            cal_dipoles_free = dipoles_for_trials

    # --- Summary ---
    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print(f"  Calibration params obtained: {list(calibrator.calibration_params.keys())}")
    print(f"  Dipole position index: {position_index}")
    print(f"  Dipole fit window: [{tmin_fit*1000:.1f}, {tmax_fit*1000:.1f}] ms")
    print(f"  Pre-stim epochs for classifier: {cal_pre_epochs.get_data(copy=False).shape}")
    print(f"  Post-stim epochs (dipole-fitted): {cal_post_epochs.get_data(copy=False).shape}")
    print("=" * 70)

    # =========================================================================
    # INTERVENTION PHASE — process post-calibration trials, fit dipoles,
    # compute normalized TEP labels using the same Calibrator preprocessing
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERVENTION PHASE — TEP label computation")
    print("=" * 70)

    n_intervention = n_total_trials - N_CALIBRATION_TRIALS
    print(f"Processing {n_intervention} intervention trials...")

    # --- Process intervention trials: require both pre & post to pass ---
    int_post_list = []
    int_pre_list = []
    for trial_idx in range(N_CALIBRATION_TRIALS, n_total_trials):
        trial = _single_trial_epochs_from_arrays(all_eeg_data, all_events, epochs, trial_idx)
        result_pre = calibrator.preprocess_pre(trial)
        result_post = calibrator.preprocess_post(trial)
        if result_pre is not False and result_post is not False:
            int_pre_list.append(result_pre)
            int_post_list.append(result_post)
        if (trial_idx - N_CALIBRATION_TRIALS + 1) % 100 == 0:
            print(f"  Preprocessed {trial_idx - N_CALIBRATION_TRIALS + 1}/{n_intervention} "
                  f"intervention trials")

    print(f"  {len(int_post_list)}/{n_intervention} intervention trials survived "
          f"(both pre & post)")
    int_post_epochs = mne.concatenate_epochs(int_post_list)

    print("\nFitting free-orientation dipoles to intervention trials...")
    int_dipoles_free, int_extraction_times = dipole_fitter.fit_trials(
        int_post_epochs, orientation=None)
    print(f"  {len(int_dipoles_free)} dipoles fitted, "
          f"avg time {np.mean(int_extraction_times)*1e3:.2f} ms")

    # --- TEP normalization (matching offline path) ---
    cal_tep_amplitudes = np.array([d['amplitude'] for d in cal_dipoles_free]).flatten()
    int_tep_amplitudes = np.array([d['amplitude'] for d in int_dipoles_free]).flatten()
    all_tep_amplitudes = np.concatenate([cal_tep_amplitudes, int_tep_amplitudes])
    period_labels = (['calibration'] * len(cal_tep_amplitudes) +
                     ['intervention'] * len(int_tep_amplitudes))

    full_metadata = pd.DataFrame({
        'TEP_amplitude': all_tep_amplitudes,
        'period': period_labels,
    })
    cal_metadata = full_metadata[full_metadata['period'] == 'calibration']

    # Fit normalizer on calibration, transform full sequence
    normalizer = TEPNormalizer(target_col='TEP_amplitude', scale_factor=1.0)
    normalizer.fit(cal_metadata)
    all_labels = normalizer.transform(full_metadata)

    # Extract intervention labels
    int_labels = all_labels[len(cal_tep_amplitudes):]
    # Also compute calibration labels (needed for calibration fine-tuning)
    cal_labels = all_labels[:len(cal_tep_amplitudes)]
    print(f"\n  Calibration TEP amplitudes: {len(cal_tep_amplitudes)}")
    print(f"  Intervention TEP amplitudes: {len(int_tep_amplitudes)}")
    print(f"  Intervention labels (first 5): {int_labels[:5]}")

    # =========================================================================
    # COMPARE WITH OFFLINE LABELS
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARE ONLINE vs OFFLINE LABELS")
    print("=" * 70)
    offline = np.load(PREDICTIONS_PATH)
    offline_labels = offline["actual_values"]

    n_compare = min(len(int_labels), len(offline_labels))
    online_labels = int_labels[:n_compare]
    offline_labels_cmp = offline_labels[:n_compare]

    print(f"  Online labels count:  {len(int_labels)}")
    print(f"  Offline labels count: {len(offline_labels)}")
    print(f"  Comparing first {n_compare} labels...")
    print(f"")
    print(f"  First online label:   {online_labels[0]:.6f}")
    print(f"  First offline label:  {offline_labels_cmp[0]:.6f}")
    print(f"  Match (first):        {np.isclose(online_labels[0], offline_labels_cmp[0], atol=1e-4)}")
    print(f"")
    diffs = np.abs(online_labels - offline_labels_cmp)
    print(f"  Max absolute diff:    {np.max(diffs):.8f}")
    print(f"  Mean absolute diff:   {np.mean(diffs):.8f}")
    print(f"  Num diffs > 0.01:     {np.sum(diffs > 0.01)}")
    print(f"  Num diffs > 0.001:    {np.sum(diffs > 0.001)}")
    match_all = np.allclose(online_labels, offline_labels_cmp, atol=1e-7)
    print(f"  All match (atol=1e-7): {match_all}")
    print("=" * 70)

    # =========================================================================
    # CLASSIFIER — Load pretrained, calibrate, run online finetuning
    # =========================================================================
    print("\n" + "=" * 70)
    print("CLASSIFIER — Pretrained model + Calibration + Online finetuning")
    print("=" * 70)

    device = torch.device("cpu")
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])

    # --- Prepare pre-stim EEG data (matching offline tmin/tmax crop) ---
    cal_pre_data = cal_pre_epochs.copy().crop(
        tmin=CONFIG["tmin"], tmax=CONFIG["tmax"], include_tmax=True
    ).get_data(copy=False)
    n_channels = cal_pre_data.shape[1]
    n_timepoints = cal_pre_data.shape[2]
    print(f"\n  Pre-stim EEG shape: ({n_channels} ch, {n_timepoints} timepoints)")
    print(f"  Calibration trials for classifier: {cal_pre_data.shape[0]}")

    # --- Build model and load pretrained weights ---
    class ArgsNamespace:
        """Minimal namespace to satisfy TTAWrapper and model building."""
        pass

    args = ArgsNamespace()
    for k, v in CONFIG.items():
        setattr(args, k, v)

    model = build_model(
        model_name="PRIME",
        n_channels=n_channels,
        n_times=n_timepoints,
        n_outputs=1,
        device=device,
        model_specific_args={},
    )

    # Load global back-rotation matrix
    global_backrot_matrix_np = np.load(GLOBAL_BACKROT_PATH)

    # Wrap model with TTA
    model_wrapped = TTAWrapper(
        model, args, sr_hz=1000.0, global_backrot_matrix_np=global_backrot_matrix_np
    ).to(device)

    # Load pretrained weights
    checkpoint = torch.load(PRETRAINED_MODEL_PATH, map_location=device, weights_only=False)
    model_wrapped.wrapped_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded pretrained model from: {PRETRAINED_MODEL_PATH}")

    # --- STAGE 2: CALIBRATION FINE-TUNING ---
    # NOTE: The offline path has a bug (missing loss.backward()) so calibration
    # training does nothing. We skip it but still initialize alignment.
    print("\n  --- Calibration (alignment init only, no training due to offline bug) ---")

    # Initialize alignment from calibration pre-stim
    model_wrapped.init_alignment_from_calibration(cal_pre_data)
    print(f"  Initialized EA alignment from {len(cal_pre_data)} calibration trials")

    # Run the calibration "training" loop to match offline behavior exactly
    # (consuming RNG state from DataLoader shuffling + optimizer state init)
    transform_np = model_wrapped.alignment_transform_torch.cpu().numpy()
    aligned_cal_data = _apply_alignment_transform_np(cal_pre_data, transform_np)
    if global_backrot_matrix_np is not None:
        aligned_cal_data = _apply_alignment_transform_np(aligned_cal_data, global_backrot_matrix_np)

    cal_epochs_t = torch.from_numpy(aligned_cal_data).float()
    cal_labels_t = torch.from_numpy(cal_labels).float()
    batch_size_cal = min(CONFIG["batch_size_finetune"], len(cal_pre_data))

    is_decision_only_mode = CONFIG["finetune_mode"] == "decision_only"
    if is_decision_only_mode:
        model_wrapped.enable_full_model_update(enabled=True)

    optimizer_calib = torch.optim.AdamW(
        model_wrapped.parameters(), lr=CONFIG["lr_calibration"]
    )
    criterion = nn.BCEWithLogitsLoss()

    # Match offline: run forward pass but no backward (reproduces the offline bug)
    model_wrapped.train()
    from torch.utils.data import DataLoader, TensorDataset

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, epochs_t, labels_t):
            self.epochs = epochs_t
            self.labels = labels_t
        def __len__(self):
            return len(self.epochs)
        def __getitem__(self, idx):
            return {"epoch": self.epochs[idx], "label": self.labels[idx]}

    calib_dataset = DictDataset(cal_epochs_t, cal_labels_t)
    calib_gen = torch.Generator()
    calib_gen.manual_seed(CONFIG["seed"])
    calib_loader = DataLoader(
        calib_dataset, batch_size=batch_size_cal, shuffle=True,
        num_workers=0, pin_memory=True, generator=calib_gen
    )

    for epoch in range(CONFIG["calibration_epochs"]):
        for batch in calib_loader:
            X_batch = batch['epoch'].to(device)
            y_batch = batch['label'].to(device).unsqueeze(1)
            optimizer_calib.zero_grad()
            logits = model_wrapped(X_batch)
            loss = criterion(logits, y_batch)
            # NOTE: no loss.backward() — matching offline bug
            optimizer_calib.step()
    model_wrapped.eval()

    if is_decision_only_mode:
        model_wrapped.enable_full_model_update(enabled=False)

    print(f"  Calibration phase complete (no actual weight update due to missing backward)")

    # --- STAGE 3: ONLINE FINE-TUNING (trial-by-trial) ---
    print("\n  --- Online fine-tuning simulation ---")

    # Reset RNG state for deterministic dropout/operations during online simulation
    torch.manual_seed(CONFIG["seed"])

    # Prepare intervention pre-stim data
    int_pre_epochs = mne.concatenate_epochs(int_pre_list)
    int_pre_data = int_pre_epochs.copy().crop(
        tmin=CONFIG["tmin"], tmax=CONFIG["tmax"], include_tmax=True
    ).get_data(copy=False)
    n_online_trials = len(int_pre_data)

    # Restrict to first 150 trials
    n_online_trials = min(n_online_trials, 150)
    
    print(f"  Online trials: {n_online_trials}")

    # Setup optimizer and buffers
    optimizer_finetune = torch.optim.AdamW(
        model_wrapped.parameters(),
        lr=CONFIG["lr_finetune"],
        weight_decay=CONFIG["weight_decay_finetune"],
    )
    max_window_size = min(CONFIG["window_size"], n_online_trials)
    epoch_buffer = deque(maxlen=max_window_size)
    label_buffer = deque(maxlen=max_window_size)

    all_predictions = []
    batch_size_ft = CONFIG["batch_size_finetune"]

    for trial_idx in range(n_online_trials):
        single_epoch_np = int_pre_data[trial_idx]
        single_label_np = int_labels[trial_idx]

        # --- PREDICT ---
        single_epoch_t = torch.from_numpy(single_epoch_np).float().unsqueeze(0).to(device)
        model_wrapped.eval()
        with torch.no_grad():
            logits = model_wrapped.predict(single_epoch_t)
        pred_prob = torch.sigmoid(logits).item()
        all_predictions.append(pred_prob)

        # --- ADAPT ALIGNMENT ---
        model_wrapped.adapt_alignment(single_epoch_np)

        # --- BUFFER for fine-tuning ---
        epoch_buffer.append(single_epoch_np)
        label_buffer.append(single_label_np)

        # --- FINE-TUNE when buffer is full ---
        if len(epoch_buffer) >= batch_size_ft:
            epochs_for_ft = np.array(epoch_buffer)
            labels_for_ft = np.array(label_buffer)

            # Apply current alignment transform + back-rotation
            transform = model_wrapped.alignment_transform_torch.cpu().numpy()
            epochs_for_ft = _apply_alignment_transform_np(epochs_for_ft, transform)
            if global_backrot_matrix_np is not None:
                epochs_for_ft = _apply_alignment_transform_np(
                    epochs_for_ft, global_backrot_matrix_np
                )

            # Create batch and train using DataLoader with seeded generator (matches offline)
            ft_epochs_t = torch.from_numpy(epochs_for_ft).float()
            ft_labels_t = torch.from_numpy(labels_for_ft).float()
            ft_dataset = DictDataset(ft_epochs_t, ft_labels_t)
            ft_gen = torch.Generator()
            ft_gen.manual_seed(CONFIG["seed"] + trial_idx)
            ft_loader = DataLoader(
                ft_dataset, batch_size=batch_size_ft, shuffle=True,
                num_workers=0, pin_memory=True, generator=ft_gen
            )

            model_wrapped.train()
            # Single epoch of fine-tuning on the buffer
            for _ in range(CONFIG["finetune_epochs"]):
                for batch in ft_loader:
                    X_batch = batch['epoch'].to(device)
                    y_batch = batch['label'].to(device).unsqueeze(1)
                    optimizer_finetune.zero_grad(set_to_none=True)
                    logits_ft = model_wrapped(X_batch, is_finetuning_batch=True)
                    loss_ft = criterion(logits_ft, y_batch)
                    loss_ft.backward()
                    optimizer_finetune.step()
            model_wrapped.eval()

        if (trial_idx + 1) % 100 == 0:
            print(f"    Trial {trial_idx + 1}/{n_online_trials}")

    all_predictions = np.array(all_predictions)
    print(f"\n  Online simulation complete. Predictions: {len(all_predictions)}")
    print(f"  First 5 predictions: {all_predictions[:5]}")

    # =========================================================================
    # COMPARE WITH OFFLINE PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARE ONLINE vs OFFLINE")
    print("=" * 70)
    offline = np.load(PREDICTIONS_PATH)
    offline_preds = offline["predictions"]
    offline_labels = offline["actual_values"]

    # --- Labels comparison ---
    n_compare = min(len(int_labels), len(offline_labels))
    label_diffs = np.abs(int_labels[:n_compare] - offline_labels[:n_compare])
    print(f"\n  LABELS ({n_compare} trials):")
    print(f"    Max diff:       {np.max(label_diffs):.2e}")
    print(f"    Num diffs > 0:  {np.sum(label_diffs > 0)}")
    print(f"    Exact match:    {np.array_equal(int_labels[:n_compare], offline_labels[:n_compare])}")

    # --- Predictions comparison ---
    n_compare_pred = min(len(all_predictions), len(offline_preds))
    pred_diffs = np.abs(all_predictions[:n_compare_pred] - offline_preds[:n_compare_pred])
    print(f"\n  PREDICTIONS ({n_compare_pred} trials):")
    print(f"    Max diff:       {np.max(pred_diffs):.6f}")
    print(f"    Mean diff:      {np.mean(pred_diffs):.6f}")
    print(f"    Num exact:      {np.sum(pred_diffs == 0)}/{n_compare_pred}")
    print(f"    Num < 1e-6:     {np.sum(pred_diffs < 1e-6)}")
    print(f"    Num < 1e-4:     {np.sum(pred_diffs < 1e-4)}")
    print(f"    Num < 0.001:    {np.sum(pred_diffs < 0.001)}")
    print(f"    All close (1e-4): {np.allclose(all_predictions[:n_compare_pred], offline_preds[:n_compare_pred], atol=1e-4)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
