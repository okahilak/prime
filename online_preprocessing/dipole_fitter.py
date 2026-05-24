"""
DipoleFitter: stateful object that fits a dipole to calibration-trial evoked data
and exposes the resulting fitting_info struct.

Usage
-----
    fitter = DipoleFitter(forward)
    fitting_info = fitter.fit(epochs)   # also stored in fitter.fitting_info
"""
import numpy as np
import pandas as pd


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


# ==================== DipoleFitter class ====================

class DipoleFitter:
    """Fits a dipole to calibration-trial evoked data.

    Parameters
    ----------
    forward : mne.Forward
        Forward solution (already channel-picked).
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

    def __init__(self, forward, tmin_init=0.038, tmax_init=0.050,
                 min_window_size=3, max_window_size=6, window_size_exponent=1.5):
        self._forward = forward
        self._tmin_init = tmin_init
        self._tmax_init = tmax_init
        self._min_window_size = min_window_size
        self._max_window_size = max_window_size
        self._window_size_exponent = window_size_exponent
        self._fitting_info = None

    def fit(self, epochs):
        """Compute dipole fitting parameters from calibration epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            Calibration epochs (post-stimulus).

        Returns
        -------
        fitting_info : dict
            Dictionary with keys ``position_index``, ``orientation``, and
            ``time_range``.  Also stored in ``self.fitting_info``.
        """
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

    @property
    def fitting_info(self):
        """Fitting parameters dict (available after ``fit()`` is called)."""
        return self._fitting_info
