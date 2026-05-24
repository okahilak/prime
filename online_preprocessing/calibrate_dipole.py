"""
Part 1: Compute dipole fitting info from calibration trials.

Uses the first N calibration trials to determine the optimal position,
orientation, and time range for single-trial dipole fitting.
Outputs: {subject}_dipole_fitting_info.npz (position_index, orientation, time_range)
"""
import mne
import numpy as np
import os
import argparse
from pathlib import Path

from dipole_fitter import DipoleFitter

DATA_ROOT = Path("~/prime-data").expanduser()

mne.set_log_level("ERROR")


def run_calibration(subject, subjects_directory_eeg, forward):
    """
    Computes dipole fitting parameters from calibration trials for a single subject.
    Saves the fitting info to an .npz file.
    """
    print(f"--- Calibration: starting for subject {subject} ---")

    subject_directory = os.path.join(subjects_directory_eeg, subject)

    try:
        epochs = mne.read_epochs(os.path.join(subject_directory, f'{subject}_calibration_post.fif'), verbose=False)
    except FileNotFoundError:
        print(f"ERROR: Could not find post-stimulus epoch file for subject {subject}. Skipping.")
        return

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


def main():
    parser = argparse.ArgumentParser(description="Compute dipole fitting info from calibration trials.")
    parser.add_argument("--subject", required=True, type=str, help="The subject identifier (e.g., 'sub-001').")
    args = parser.parse_args()

    subjects_directory_eeg = str(DATA_ROOT / "processed")
    fsaverage_forward_path = os.path.join(DATA_ROOT / "fsaverage", "fsaverage-fwd.fif")

    common_channels = [
        'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
        'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2',
        'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4',
        'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1',
        'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3',
        'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8'
    ]

    os.makedirs(subjects_directory_eeg, exist_ok=True)

    forward = mne.read_forward_solution(fsaverage_forward_path, verbose=False)
    forward = forward.pick_channels(common_channels, ordered=True)

    run_calibration(
        subject=args.subject,
        subjects_directory_eeg=subjects_directory_eeg,
        forward=forward,
    )


if __name__ == "__main__":
    main()
