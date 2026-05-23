#%%
"""
Core dipole fitting functions: scanning, time-range selection, and single-trial extraction.
"""
import mne 
import numpy as np
from sklearn.metrics import r2_score
import time 
import pandas as pd

mne.set_log_level("ERROR")


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
    
	return time_range, dipoles_in_time_range, df


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

def determine_optimal_ori_and_pos(dipoles, forward, evoked):
    #get the dipole amplitudes, orientations and positions
    dipole_amplitudes = np.array([dipole['amplitude'] for dipole in dipoles])
    dipole_orientations = np.array([dipole['orientation'] for dipole in dipoles])
    dipole_positions = np.array([dipole['position'] for dipole in dipoles])
    dipole_times = np.array([dipole['time'] for dipole in dipoles])
    #get weighted dipole position and orientation measures
    weighted_ori = np.average(dipole_orientations, weights=dipole_amplitudes, axis=0)
    weighted_ori /= np.linalg.norm(weighted_ori) #ensure that the orientation is unit-length
    weighted_pos = np.average(dipole_positions, weights=dipole_amplitudes, axis=0) #weighted dipole position
   
    #find the nearest valid position and position index to the weighted position
    pos_index = np.argmin(np.linalg.norm(forward['source_rr'] - weighted_pos, axis=1))
    pos_of_pos_index = forward['source_rr'][pos_index] #position of the nearest valid position index

    #determine the properties of the weighted dipole
    weighted_dipole_stats_fixed = {'dipole_moments':[], 'dipole_amplitudes': [], 'data_predicteds':[], 'data_measureds':[], 'r2_scores': [], 'times': [], 'orientation':weighted_ori, 'position':weighted_pos, 'scalar_dipole_amplitudes': []}
    weighted_dipole_stats_free = {'dipole_moments':[], 'dipole_amplitudes': [], 'data_predicteds':[], 'data_measureds':[], 'r2_scores': [], 'times': [], 'orientations':[], 'position':weighted_pos}
    evoked_cropped = evoked.copy().crop(np.min(dipole_times), np.max(dipole_times))
    evoked_data = evoked_cropped.data
    L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0)
    position_now = pos_index*3
    leadfield_at_pos = L[:,position_now:position_now+3] #leadfield at position
    leadfield_in_ori = np.matmul(leadfield_at_pos, weighted_ori) #project to orientation
    leadfield_at_pos_pinv = np.linalg.pinv(leadfield_at_pos)

    for time_index in range(evoked_data.shape[1]): #get the dipole stats in the average response
        data_measured = evoked_data[:, time_index]
        amplitude = np.dot(leadfield_in_ori.T,data_measured) / np.dot(leadfield_in_ori.T, leadfield_in_ori)
        dipole_moment = weighted_ori*amplitude #Q = aq
        data_predicted = np.matmul(leadfield_at_pos, dipole_moment) #y_predicted = LQ
        #update dictionary
        weighted_dipole_stats_fixed['r2_scores'].append(r2_score(data_measured, data_predicted)) #coefficient of determination of the dipole fit
        weighted_dipole_stats_fixed['dipole_moments'].append(dipole_moment)
        weighted_dipole_stats_fixed['scalar_dipole_amplitudes'].append(amplitude)
        weighted_dipole_stats_fixed['dipole_amplitudes'].append(np.abs(amplitude))
        weighted_dipole_stats_fixed['data_predicteds'].append(data_predicted)
        weighted_dipole_stats_fixed['data_measureds'].append(data_measured)
        weighted_dipole_stats_fixed['times'].append(evoked_cropped.times[time_index])

    for time_index in range(evoked_data.shape[1]): #get the dipole stats in the average response for free ori
        data_measured = evoked_data[:, time_index]
        dipole_moment = np.matmul(leadfield_at_pos_pinv,data_measured)
        data_predicted = np.matmul(leadfield_at_pos, dipole_moment) #y_predicted = LQ
        amplitude = np.linalg.norm(dipole_moment)
        #update dictionary
        weighted_dipole_stats_free['r2_scores'].append(r2_score(data_measured, data_predicted)) #coefficient of determination of the dipole fit
        weighted_dipole_stats_free['dipole_moments'].append(dipole_moment)
        weighted_dipole_stats_free['dipole_amplitudes'].append(amplitude)
        weighted_dipole_stats_free['orientations'].append(dipole_moment/amplitude)
        weighted_dipole_stats_free['data_predicteds'].append(data_predicted)
        weighted_dipole_stats_free['data_measureds'].append(data_measured)
        weighted_dipole_stats_free['times'].append(evoked_cropped.times[time_index])
   
    return weighted_pos, weighted_ori, pos_index, pos_of_pos_index, weighted_dipole_stats_fixed, weighted_dipole_stats_free

def init_dipole_params():
	return -np.inf, None, None, None
