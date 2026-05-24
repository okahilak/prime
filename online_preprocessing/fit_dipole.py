"""
Dipole calibration and single-trial fitting pipeline.

Runs calibration then fits dipoles to all trials for one subject.
Outputs: {subject}_dipole_fitting_info.npz
         {subject}_calibration_dipoles.npz
         {subject}_intervention_dipoles.npz
"""
import mne
import numpy as np
import os
import argparse
from pathlib import Path

from dipole_fitter import DipoleFitter

DATA_ROOT = Path("~/prime-data").expanduser()

mne.set_log_level("ERROR")

COMMON_CHANNELS = [
    'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2',
    'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4',
    'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1',
    'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3',
    'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8'
]


def run_calibration(subject, subjects_directory_eeg, forward):
    """
    Computes dipole fitting parameters from calibration trials for a single subject.
    Saves the fitting info to an .npz file and returns the fitted DipoleFitter.
    """
    print(f"--- Calibration: starting for subject {subject} ---")

    subject_directory = os.path.join(subjects_directory_eeg, subject)

    try:
        epochs = mne.read_epochs(os.path.join(subject_directory, f'{subject}_calibration_post.fif'), verbose=False)
    except FileNotFoundError:
        print(f"ERROR: Could not find post-stimulus epoch file for subject {subject}. Skipping.")
        return None

    if not epochs.info['ch_names'] == forward.ch_names:
        raise ValueError(f"Channel mismatch for subject {subject}. Aborting.")

    fitter = DipoleFitter(forward)
    fitting_info = fitter.fit(epochs)

    # Save
    os.makedirs(subject_directory, exist_ok=True)
    output_path = os.path.join(subject_directory, f'{subject}_dipole_fitting_info.npz')
    np.savez(output_path, **fitting_info)
    print(f"Fitting info saved to {output_path}")
    print(f"--- Calibration: finished for subject {subject} ---")
    return fitter


def run_fitting(subject, subjects_directory_eeg, forward, fitter):
    """
    Fits dipoles to all single trials using a calibrated DipoleFitter.
    Saves the fitted dipoles to an .npz file.
    """
    print(f"--- Fitting: starting for subject {subject} ---")

    subject_directory = os.path.join(subjects_directory_eeg, subject)

    for group_label in ('calibration', 'intervention'):
        epoch_path = os.path.join(subject_directory, f'{subject}_{group_label}_post.fif')
        try:
            epochs = mne.read_epochs(epoch_path, verbose=False)
        except FileNotFoundError:
            print(f"ERROR: Could not find {epoch_path}. Skipping {group_label}.")
            continue

        if not epochs.info['ch_names'] == forward.ch_names:
            raise ValueError(f"Channel mismatch for subject {subject} ({group_label}). Aborting.")

        fitted_dipoles = {}
        for orientation_label, orientation in [('fixed_ori', 'use_fitted'), ('free_ori', None)]:
            dipoles_for_trials, extraction_times = fitter.fit_trials(epochs, orientation=orientation)
            fitted_dipoles[f'trial_dipoles_{orientation_label}'] = dipoles_for_trials
            print(f"[{group_label}] Average {orientation_label} dipole extraction time: {np.mean(extraction_times)*1e3:.2f} ms")

        os.makedirs(subject_directory, exist_ok=True)
        output_path = os.path.join(subject_directory, f'{subject}_{group_label}_dipoles.npz')
        np.savez(output_path, **fitted_dipoles)
        print(f"Fitted dipoles saved to {output_path}")

    print(f"--- Fitting: finished for subject {subject} ---")


def main():
    parser = argparse.ArgumentParser(description="Dipole calibration and fitting pipeline.")
    parser.add_argument("--subject", required=True, type=str, help="Subject identifier (e.g. 'sub-001').")
    args = parser.parse_args()

    subjects_directory_eeg = str(DATA_ROOT / "processed")
    fsaverage_forward_path = os.path.join(DATA_ROOT / "fsaverage", "fsaverage-fwd.fif")

    os.makedirs(subjects_directory_eeg, exist_ok=True)
    forward = mne.read_forward_solution(fsaverage_forward_path, verbose=False)
    forward = forward.pick_channels(COMMON_CHANNELS, ordered=True)

    fitter = run_calibration(subject=args.subject, subjects_directory_eeg=subjects_directory_eeg, forward=forward)
    if fitter is not None:
        run_fitting(subject=args.subject, subjects_directory_eeg=subjects_directory_eeg, forward=forward, fitter=fitter)


if __name__ == "__main__":
    main()

