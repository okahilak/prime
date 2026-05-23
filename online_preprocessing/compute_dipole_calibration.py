"""
Part 1: Compute dipole fitting info from calibration trials.

Uses the first N calibration trials to determine the optimal position,
orientation, and time range for single-trial dipole fitting.
Outputs: {subject}_dipole_fitting_info.npz
"""
import mne
import numpy as np
import os
import argparse
from pathlib import Path

from compute_dipole import (
    dipoles_for_times,
    determine_optimal_time_range,
    determine_optimal_ori_and_pos,
)

DATA_ROOT = Path("~/prime-data").expanduser()

mne.set_log_level("ERROR")


def run_calibration(subject, subjects_directory_eeg, forward):
    """
    Computes dipole fitting parameters from calibration trials for a single subject.
    Saves the fitting info to an .npz file.
    """
    print(f"--- Calibration: starting for subject {subject} ---")

    tmin_init, tmax_init = 0.038, 0.050
    min_window_size = 3
    max_window_size = 6
    window_size_exponent = 1.5
    n_calibration_trials = 100

    subject_directory = os.path.join(subjects_directory_eeg, subject)

    try:
        epochs = mne.read_epochs(os.path.join(subject_directory, f'{subject}_post.fif'), verbose=False)
    except FileNotFoundError:
        print(f"ERROR: Could not find post-stimulus epoch file for subject {subject}. Skipping.")
        return

    if not epochs.info['ch_names'] == forward.ch_names:
        raise ValueError(f"Channel mismatch for subject {subject}. Aborting.")

    evoked = epochs.copy()[:n_calibration_trials].average()

    # Scan all candidate dipole positions in the initial time window
    best_dipole_per_time = dipoles_for_times(evoked, forward, tmin_init, tmax_init)

    # Determine optimal fitting time range
    optimal_time_range, dipoles_in_time_range, windows_df = determine_optimal_time_range(
        best_dipole_per_time, min_window_size, max_window_size, window_size_exponent
    )

    best_dipole_index = np.argmax([dipole['r2'] for dipole in dipoles_in_time_range])

    # Determine optimal position and orientation
    weighted_pos, weighted_ori, position_index, pos_of_pos_index, weighted_dipole_stats_fixed, weighted_dipole_stats_free = determine_optimal_ori_and_pos(
        dipoles_in_time_range, forward, evoked
    )

    # Assemble fitting info
    fitting_info = {
        'best_dipole_per_time': best_dipole_per_time,
        'optimal_time_range': optimal_time_range,
        'dipoles_in_time_range': dipoles_in_time_range,
        'windows_df': windows_df,
        'best_dipole_to_evoked': dipoles_in_time_range[best_dipole_index],
        'weighted_pos': weighted_pos,
        'weighted_ori': weighted_ori,
        'nearest_to_weighted_pos_pos_index': position_index,
        'pos_of_weighted_pos_index': pos_of_pos_index,
        'weighted_dipole_stats_fixed': weighted_dipole_stats_fixed,
        'weighted_dipole_stats_free': weighted_dipole_stats_free,
    }

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
