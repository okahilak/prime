"""
Part 2: Fit dipoles to single trials using pre-computed fitting info.

Loads the fitting info produced by calibrate_dipole.py and fits
dipoles to every trial in the epoch file.
Outputs: {subject}_calibration_dipoles.npz, {subject}_intervention_dipoles.npz
"""
import mne
import numpy as np
import os
import argparse
import time
from pathlib import Path
from sklearn.metrics import r2_score

DATA_ROOT = Path("~/prime-data").expanduser()

mne.set_log_level("ERROR")


def fit_dipoles_to_single_trials(epochs, forward, position_index, orientation, tmin, tmax):
	epochs_cropped = epochs.copy().crop(tmin, tmax) #epochs in the time range of interest
	epochs_data = epochs_cropped.get_data(copy=True) #epoched data in the time range of interest
	n_trials = epochs_data.shape[0] #number of trials in the data
	L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0) #leadfield in average reference
	position_now = position_index*3
	leadfield_at_pos = L[:,position_now:position_now+3] #leadfield of the current position

	dipoles_for_trials = [] #initialize a list for adding dipoles for each trial to

	if orientation is not None:
		orientation_is_fixed = True
		leadfield_in_ori = np.matmul(leadfield_at_pos, orientation) #project to orientation
		leadfield_at_pos_pinv = None #no pinv needed
	else:
		orientation_is_fixed = False
		leadfield_in_ori = None
		leadfield_at_pos_pinv = np.linalg.pinv(leadfield_at_pos)

	extraction_times = [] #init list for saving extraction times

	for trial_index in range(n_trials):
		#simulate getting a single-trial epoch
		epoch_now = epochs_cropped[trial_index]
		trial_data = epoch_now.get_data(copy=True)[0,:,:] #data n_channels x n_times for the single trial data
		start_time = time.perf_counter()
		best_y_predicted, best_amplitude, ori, best_dipole_moment, best_time, best_r2, best_data_measured = get_single_trial_dipole(trial_data, trial_index, epoch_now.times, orientation,
																												 leadfield_in_ori, leadfield_at_pos, leadfield_at_pos_pinv)

		true_amplitude = np.abs(best_amplitude) if orientation is not None else best_amplitude #save magnitude and scalar amplitude separately if needed

		dipole_info = {'amplitude':true_amplitude, 'dipole':best_dipole_moment, 'time': best_time, 'position': forward['source_rr'][position_index],
				  'position_index': position_index, 'position_now': position_now, 'orientation': ori, 'r2': best_r2, 'orientation_is_fixed':orientation_is_fixed,
					'y_predicted': best_y_predicted, 'y_measured': best_data_measured, 'trial_index':trial_index,
					 'leadfield_at_pos':leadfield_at_pos, 'leadfield_at_pos_pinv':leadfield_at_pos_pinv, 'leadfield_in_ori':leadfield_in_ori}
		
		if orientation is not None: #then also save the scalar amplitude
			dipole_info['scalar_amplitude'] = best_amplitude

		dipoles_for_trials.append(dipole_info) #add the information to the list
		extraction_times.append(time.perf_counter() - start_time) #append the extraction (and saving) time

	return dipoles_for_trials, extraction_times


def get_single_trial_dipole(trial_data, trial_index, times, orientation, leadfield_in_ori, leadfield_at_pos, leadfield_at_pos_pinv):
	best_r2 = -np.inf #initialize best r2 value
	best_r2_default = best_r2

	for time_index, time in enumerate(times):

		data_measured = trial_data[:, time_index] #get the data at the current time point

		if orientation is not None:
			#scalar amplitude of the fixed source, from y_measured = amplitude * leadfield_in_ori = leadfield_in_ori * amplitude
			amplitude = np.dot(leadfield_in_ori.T,data_measured) /  np.dot(leadfield_in_ori.T, leadfield_in_ori)
			dipole_moment = orientation*amplitude #Q = aq
			ori = orientation
		else:
			dipole_moment = np.matmul(leadfield_at_pos_pinv,data_measured) #Q = pinv(L(r))y
			amplitude = np.linalg.norm(dipole_moment) #a = ||Q|| (l2-norm)
			ori = dipole_moment/amplitude #ori = Q/||Q||

		data_predicted = np.matmul(leadfield_at_pos, dipole_moment) #y_predicted = LQ

		r2_now = r2_score(data_measured, data_predicted) #coefficient of determination of the dipole fit
		
		if r2_now > best_r2: #then update the best dipole statistics
			best_r2 = r2_now #update the best r2 that must be exceeded to update the best dipole statistic
			best_y_predicted = data_predicted #update the best predicted topography (forward-modeled dipole)
			best_amplitude = amplitude #update the best amplitude
			best_dipole_moment = dipole_moment #update best dipole moment
			best_time = time #update the best time
			best_data_measured = data_measured #update the respectively "measured data"

	if best_r2 == best_r2_default:
		raise ValueError(f"Did not find a sufficient R2 for trial {trial_index}")

	return best_y_predicted, best_amplitude, ori, best_dipole_moment, best_time, best_r2, best_data_measured


def init_dipole_params():
	return -np.inf, None, None, None


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
        print(f"ERROR: Fitting info not found at {fitting_info_path}. Run calibrate_dipole.py first.")
        return

    fitting_info = np.load(fitting_info_path, allow_pickle=True)
    position_index = int(fitting_info['position_index'])
    orientation = fitting_info['orientation']
    time_range = fitting_info['time_range']
    tmin, tmax = float(time_range[0]), float(time_range[1])

    # Fit and save dipoles separately for calibration and intervention trials
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
        for fixed_orientation in [orientation, None]:
            dipoles_for_trials, extraction_times = fit_dipoles_to_single_trials(
                epochs, forward, position_index, fixed_orientation, tmin, tmax
            )
            orientation_identifier = 'free_ori' if fixed_orientation is None else 'fixed_ori'
            fitted_dipoles[f'trial_dipoles_{orientation_identifier}'] = dipoles_for_trials
            print(f"[{group_label}] Average {orientation_identifier} dipole extraction time: {np.mean(extraction_times)*1e3:.2f} ms")

        os.makedirs(subject_directory, exist_ok=True)
        output_path = os.path.join(subject_directory, f'{subject}_{group_label}_dipoles.npz')
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
