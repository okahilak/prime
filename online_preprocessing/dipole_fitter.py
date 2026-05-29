"""
DipoleFitter: stateful object that calibrates dipole fitting parameters from evoked
data and fits dipoles to individual trials.

Usage
-----
    fitter = DipoleFitter(forward_path)
    fitting_info = fitter.fit(trials)       # also stored in fitter.fitting_info
    dipoles = fitter.fit_trials(trials)              # fixed orientation
    dipoles = fitter.fit_trials(trials, orientation=None)  # free orientation
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import mne


def dipoles_for_times(evoked, forward, tmin, tmax):
	evoked = evoked.copy().crop(tmin, tmax) #crop the data to the time range of interest
	data_measured = evoked.data #data in the evoked object
	L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0) #leadfield in average reference
	data_measured = data_measured - np.mean(data_measured, axis=0) #data ensured to be in average reference (if already not)
	n_pos = L.shape[1] // 3  #number of candidate source positions
	best_dipole_per_time = [] #initialize a list for optimal dipoles for each time point

	# Batch-precompute all pseudoinverses once.
	# Reshape L (n_ch, n_pos*3) -> (n_pos, n_ch, 3), then pinv -> (n_pos, 3, n_ch).
	# This replaces n_pos sequential np.linalg.pinv calls with a single batched LAPACK call.
	L_3d = L.reshape(L.shape[0], n_pos, 3).transpose(1, 0, 2)  # (n_pos, n_ch, 3)
	pinv_all = np.linalg.pinv(L_3d)                              # (n_pos, 3, n_ch)

	for time_index, time in enumerate(evoked.times): #go through the time range of interest
		y = data_measured[:, time_index]
		ss_tot = np.dot(y, y)

		# Vectorized dipole fitting over all source positions simultaneously
		Q_all = pinv_all @ y                                          # (n_pos, 3): Q = pinv(L(r)) y
		ypred_all = np.einsum('nij,nj->ni', L_3d, Q_all)             # (n_pos, n_ch): ypred = L(r) Q
		# R2 = ||ypred||^2 / ||y||^2  (valid since both are average-referenced -> zero mean)
		R2_all = np.einsum('ni,ni->n', ypred_all, ypred_all) / ss_tot # (n_pos,)

		best_pos_index = int(np.argmax(R2_all))
		best_r2 = float(R2_all[best_pos_index])
		best_dipole_moment = Q_all[best_pos_index]
		best_data_predicted = ypred_all[best_pos_index]

		#compute useful information for the best dipole now that all candidates have been considered
		amplitude = np.linalg.norm(best_dipole_moment) #dipole amplitude is the norm of the dipole moment
		orientation = best_dipole_moment / amplitude #unit-length dipole orientation is the dipole moment divided by amplitude
		position = forward['source_rr'][best_pos_index] #position of the dipole in the source space

		#store the statistics of the best dipole to a dictionary
		best_dipole_statistics = {'amplitude': amplitude, 'orientation':orientation, 'position': position, 'time':time,
						'position_index':best_pos_index, 'moment': best_dipole_moment, 'r2':best_r2, 'data_predicted': best_data_predicted}

		best_dipole_per_time.append(best_dipole_statistics) #store the best dipole of the current time to the list

	return best_dipole_per_time #return the list of dipoles in dictionaries for each time point


def determine_optimal_time_range(dipoles, min_window_size, max_window_size, window_size_exponent):
	n_times = len(dipoles)
	amplitudes = np.array([dipole['amplitude'] for dipole in dipoles])
	positions = np.array([dipole['position'] for dipole in dipoles])
	orientations = np.array([dipole['orientation'] for dipole in dipoles])
	if max_window_size is None:
		max_window_size = n_times

	# --- 1. Pre-computation loop ---
	window_data = []
	for window_size in range(min_window_size, max_window_size + 1):
		for start in range(n_times - window_size + 1):
			end = start + window_size

			#Amplitude
			amplitude_score = np.min(amplitudes[start:end])
            
            # Position stability
			positions_in_window = positions[start:end]
			mean_pos = np.mean(positions_in_window, axis=0)
			pos_stability = np.mean(np.linalg.norm(positions_in_window - mean_pos, axis=1))

            # Orientation stability
			orientations_in_window = orientations[start:end]
			mean_ori = np.mean(orientations_in_window, axis=0)
			mean_ori /= np.linalg.norm(mean_ori)
			dot_products = np.dot(orientations_in_window, mean_ori)
			ori_stability = np.mean(np.degrees(np.arccos(np.clip(dot_products, -1.0, 1.0))))
            
			window_data.append({
                'start': start, 'end': end, 'size': window_size,
                'pos_stability': pos_stability, 'ori_stability': ori_stability, 'min_amplitude':amplitude_score, 'time_range': (dipoles[start]['time'], dipoles[end - 1]['time']),
            })

    # --- 2. Convert to DataFrame and Rank --
	df = pd.DataFrame(window_data)
    # ranks in descending or ascending order
	df['min_amplitude_rank'] = df['min_amplitude'].rank(method='average', ascending=False)
	df['size_rank'] = df['size'].rank(method='average', ascending=True)
	df['pos_rank'] = df['pos_stability'].rank(method='average', ascending=True)
	df['ori_rank'] = df['ori_stability'].rank(method='average', ascending=True)

	df['combined_deviation_rank'] = df['pos_rank'] + df['ori_rank'] #combined deviation rank

    # --- 3. Calculate final segment score ---
    # Score rewards window size and amplitude and penalizes high (bad) combined deviation rank
	df['final_score'] = (df['size_rank']**(window_size_exponent)) / (df['combined_deviation_rank'] + df['min_amplitude_rank'])

    # --- 4. Find the best segment ---
	best_segment_row = df.loc[df['final_score'].idxmax()]
	start, end = int(best_segment_row['start']), int(best_segment_row['end'])

	time_range = (dipoles[start]['time'], dipoles[end - 1]['time'])
	dipoles_in_time_range = dipoles[start:end]
    
	return time_range, dipoles_in_time_range


def determine_optimal_ori_and_pos(dipoles, forward):
    dipole_amplitudes = np.array([dipole['amplitude'] for dipole in dipoles])
    dipole_orientations = np.array([dipole['orientation'] for dipole in dipoles])
    dipole_positions = np.array([dipole['position'] for dipole in dipoles])
    weighted_ori = np.average(dipole_orientations, weights=dipole_amplitudes, axis=0)
    weighted_ori /= np.linalg.norm(weighted_ori)
    weighted_pos = np.average(dipole_positions, weights=dipole_amplitudes, axis=0)
    position_index = int(np.argmin(np.linalg.norm(forward['source_rr'] - weighted_pos, axis=1)))
    return position_index, weighted_ori


# ==================== Single-trial fitting ====================

def fit_dipoles_to_single_trials(epochs, forward, position_index, orientation, tmin, tmax):
	epochs_cropped = epochs.copy().crop(tmin, tmax) #epochs in the time range of interest
	epochs_data = epochs_cropped.get_data(copy=True) #epoched data in the time range of interest
	n_trials = epochs_data.shape[0] #number of trials in the data
	L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0) #leadfield in average reference
	position_now = position_index*3
	leadfield_at_pos = L[:,position_now:position_now+3] #leadfield of the current position

	amplitudes = np.empty(n_trials)

	if orientation is not None:
		leadfield_in_ori = np.matmul(leadfield_at_pos, orientation) #project to orientation
		leadfield_at_pos_pinv = None #no pinv needed
	else:
		leadfield_in_ori = None
		leadfield_at_pos_pinv = np.linalg.pinv(leadfield_at_pos)

	for trial_index in range(n_trials):
		epoch_now = epochs_cropped[trial_index]
		trial_data = epoch_now.get_data(copy=True)[0,:,:] #data n_channels x n_times for the single trial data
		_, best_amplitude, _, _, _, _, _ = get_single_trial_dipole(trial_data, trial_index, epoch_now.times, orientation,
																	 leadfield_in_ori, leadfield_at_pos, leadfield_at_pos_pinv)

		amplitudes[trial_index] = np.abs(best_amplitude) if orientation is not None else best_amplitude

	return amplitudes


def get_single_trial_dipole(trial_data, trial_index, times, orientation, leadfield_in_ori, leadfield_at_pos, leadfield_at_pos_pinv):
	best_r2 = -np.inf #initialize best r2 value
	best_r2_default = best_r2

	for time_index, time_now in enumerate(times):
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
			best_time = time_now #update the best time
			best_data_measured = data_measured #update the respectively "measured data"

	if best_r2 == best_r2_default:
		raise ValueError(f"Did not find a sufficient R2 for trial {trial_index}")

	return best_y_predicted, best_amplitude, ori, best_dipole_moment, best_time, best_r2, best_data_measured


# ==================== DipoleFitter class ====================

COMMON_CHANNELS = [
    'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz', 'F1', 'F2',
    'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4',
    'FC5', 'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1',
    'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3',
    'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8'
]


class DipoleFitter:
    """Fits a dipole to calibration-trial evoked data.

    Parameters
    ----------
    forward_path : str
        Path to the forward solution.
    tmin_init : float
        Start of the initial search window (seconds).
    tmax_init : float
        End of the initial search window (seconds).
    min_window_size : int
        Minimum window size (time samples) for ``determine_optimal_time_range``.
    max_window_size : int or None
        Maximum window size.  ``None`` means unconstrained.
    window_size_exponent : float
        Exponent for window size scoring.
    """

    def __init__(self, forward_path, tmin_init=0.038, tmax_init=0.050,
                 min_window_size=3, max_window_size=6, window_size_exponent=1.5):
        self._forward = mne.read_forward_solution(str(forward_path), verbose=False)

		# TODO: Is this necessary? Note that it's not done in the calibrator.
        self._forward = self._forward.pick_channels(COMMON_CHANNELS, ordered=True)

        self._tmin_init = tmin_init
        self._tmax_init = tmax_init
        self._min_window_size = min_window_size
        self._max_window_size = max_window_size
        self._window_size_exponent = window_size_exponent
        self._fitting_info = None

    def _epochs_from_trials(self, trials):
        """Extract concatenated post-stim epochs from a list of ProcessedTrial or pass through mne.Epochs."""
        if isinstance(trials, mne.BaseEpochs):
            return trials
        return mne.concatenate_epochs([t.epoch_post for t in trials])

    def fit(self, trials):
        """Compute dipole fitting parameters from calibration trials.

        Parameters
        ----------
        trials : list of ProcessedTrial or mne.Epochs
            Calibration trials (post-stimulus epochs are used).

        Returns
        -------
        fitting_info : dict
            Dictionary with keys ``position_index``, ``orientation``, and
            ``time_range``.  Also stored in ``self.fitting_info``.
        """
        epochs = self._epochs_from_trials(trials)
        if not epochs.info['ch_names'] == self._forward.ch_names:
            raise ValueError(f"Channel mismatch for epochs and forward solution. Aborting.")

        evoked = epochs.copy().average()
        best_dipole_per_time = dipoles_for_times(
            evoked, self._forward, self._tmin_init, self._tmax_init
        )
        time_range, dipoles_in_time_range = determine_optimal_time_range(
            best_dipole_per_time, self._min_window_size,
            self._max_window_size, self._window_size_exponent
        )
        position_index, orientation = determine_optimal_ori_and_pos(
            dipoles_in_time_range, self._forward
        )
        self._fitting_info = {
            'position_index': position_index,
            'orientation': orientation,
            'time_range': time_range,
        }
        return self._fitting_info

    def fit_trials(self, trials, orientation='use_fitted'):
        """Fit dipoles to individual trials using stored fitting_info.

        Must be called after ``fit()`` or constructed via ``from_fitting_info()``.

        Parameters
        ----------
        trials : list of ProcessedTrial or mne.Epochs
            Trials to fit (post-stimulus epochs are used).
        orientation : np.ndarray, None, or 'use_fitted'
            Dipole orientation. ``'use_fitted'`` (default) uses the orientation
            stored in ``fitting_info``. ``None`` performs free-orientation fitting.

        Returns
        -------
        amplitudes : np.ndarray
            1-D array of dipole amplitudes, one per trial.
        """
        epochs = self._epochs_from_trials(trials)
        if not epochs.info['ch_names'] == self._forward.ch_names:
            raise ValueError(f"Channel mismatch for epochs and forward solution. Aborting.")

        if self._fitting_info is None:
            raise RuntimeError(
                "fitting_info must be set before calling fit_trials(). "
                "Call fit() or use from_fitting_info()."
            )
        if orientation == 'use_fitted':
            orientation = self._fitting_info['orientation']
        position_index = self._fitting_info['position_index']
        tmin, tmax = self._fitting_info['time_range']
        return fit_dipoles_to_single_trials(
            epochs, self._forward, position_index, orientation, tmin, tmax
        )

    @property
    def fitting_info(self):
        """Fitting parameters dict (available after ``fit()`` is called)."""
        return self._fitting_info
