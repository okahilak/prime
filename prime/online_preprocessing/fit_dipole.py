"""
Dipole calibration and single-trial fitting pipeline.

Runs calibration then fits dipoles to all trials for one subject.
Outputs: {subject}_calibration_amplitudes.npy
         {subject}_intervention_amplitudes.npy
"""
import mne
import numpy as np
import os
import argparse
from pathlib import Path

try:
    from .dipole_fitter import DipoleFitter
except ImportError:
    from dipole_fitter import DipoleFitter

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

mne.set_log_level("ERROR")


def run_fitting(subject, subjects_directory_eeg, forward_path):
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

    fitter = DipoleFitter(forward_path)
    _ = fitter.calibrate(epochs)

    subject_directory = os.path.join(subjects_directory_eeg, subject)

    for group_label in ('calibration', 'intervention'):
        epoch_path = os.path.join(subject_directory, f'{subject}_{group_label}_post.fif')
        try:
            epochs = mne.read_epochs(epoch_path, verbose=False)
        except FileNotFoundError:
            print(f"ERROR: Could not find {epoch_path}. Skipping {group_label}.")
            continue

        amplitudes = np.array([fitter.fit_trial(epochs[i]) for i in range(len(epochs))])

        os.makedirs(subject_directory, exist_ok=True)
        output_path = os.path.join(subject_directory, f'{subject}_{group_label}_amplitudes.npy')
        np.save(output_path, amplitudes)
        print(f"Amplitudes saved to {output_path}")

    print(f"--- Fitting: finished for subject {subject} ---")


def main():
    parser = argparse.ArgumentParser(description="Dipole calibration and fitting pipeline.")
    parser.add_argument("--subject", required=True, type=str, help="Subject identifier (e.g. 'sub-001').")
    args = parser.parse_args()

    subjects_directory_eeg = str(DATA_ROOT / "processed")
    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
    os.makedirs(subjects_directory_eeg, exist_ok=True)

    run_fitting(subject=args.subject, subjects_directory_eeg=subjects_directory_eeg, forward_path=forward_path)


if __name__ == "__main__":
    main()

