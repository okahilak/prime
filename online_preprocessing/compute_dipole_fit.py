"""
Part 2: Fit dipoles to single trials using pre-computed fitting info.

Loads the fitting info produced by compute_dipole_calibration.py and fits
dipoles to every trial in the epoch file.
Outputs: {subject}_fitted_dipoles.npz
"""
import mne
import numpy as np
import os
import argparse
from pathlib import Path

from compute_dipole import fit_dipoles_to_single_trials

DATA_ROOT = Path("~/prime-data").expanduser()

mne.set_log_level("ERROR")


def run_fitting(subject, subjects_directory_eeg, forward):
    """
    Loads pre-computed fitting info and fits dipoles to all single trials.
    Saves the fitted dipoles to an .npz file.
    """
    print(f"--- Fitting: starting for subject {subject} ---")

    subject_directory = os.path.join(subjects_directory_eeg, subject)

    # Load fitting info from calibration step
    fitting_info_path = os.path.join(subject_directory, f'{subject}_dipole_fitting_info.npz')
    if not os.path.exists(fitting_info_path):
        print(f"ERROR: Fitting info not found at {fitting_info_path}. Run compute_dipole_calibration.py first.")
        return

    fitting_info = np.load(fitting_info_path, allow_pickle=True)
    position_index = int(fitting_info['nearest_to_weighted_pos_pos_index'])
    weighted_ori = fitting_info['weighted_ori']
    optimal_time_range = fitting_info['optimal_time_range']
    tmin, tmax = float(optimal_time_range[0]), float(optimal_time_range[1])

    # Load epochs
    try:
        epochs = mne.read_epochs(os.path.join(subject_directory, f'{subject}_post.fif'), verbose=False)
    except FileNotFoundError:
        print(f"ERROR: Could not find post-stimulus epoch file for subject {subject}. Skipping.")
        return

    if not epochs.info['ch_names'] == forward.ch_names:
        raise ValueError(f"Channel mismatch for subject {subject}. Aborting.")

    # Fit dipoles with fixed and free orientation
    fitted_dipoles = {}
    for fixed_orientation in [weighted_ori, None]:
        dipoles_for_trials, extraction_times = fit_dipoles_to_single_trials(
            epochs, forward, position_index, fixed_orientation, tmin, tmax
        )
        orientation_identifier = 'free_ori' if fixed_orientation is None else 'fixed_ori'
        fitted_dipoles[f'trial_dipoles_{orientation_identifier}'] = dipoles_for_trials
        print(f"Average {orientation_identifier} dipole extraction time: {np.mean(extraction_times)*1e3:.2f} ms")

    # Save
    os.makedirs(subject_directory, exist_ok=True)
    output_path = os.path.join(subject_directory, f'{subject}_fitted_dipoles.npz')
    np.savez(output_path, **fitted_dipoles)
    print(f"Fitted dipoles saved to {output_path}")
    print(f"--- Fitting: finished for subject {subject} ---")


def main():
    parser = argparse.ArgumentParser(description="Fit dipoles to single trials using pre-computed fitting info.")
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

    run_fitting(
        subject=args.subject,
        subjects_directory_eeg=subjects_directory_eeg,
        forward=forward,
    )


if __name__ == "__main__":
    main()
