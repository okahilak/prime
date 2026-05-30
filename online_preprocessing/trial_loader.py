"""Provides raw EEG trials by index for a given subject."""

import os
from pathlib import Path

import mne

DATA_ROOT = Path("~/prime-data").expanduser()

COMMON_CHANNELS = [
    'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
    'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
    'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz',
    'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
    'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz',
    'T7', 'T8', 'TP7', 'TP8',
]


class TrialLoader:
    """Loads all epochs for a subject and provides raw trials by index."""

    def __init__(self, subject_id_string):
        epochs = self._load_subject_epochs(subject_id_string)
        self._eeg_data = epochs.get_data(copy=False)
        self._events = epochs.events
        self._epochs = epochs

    @property
    def num_trials(self):
        return self._eeg_data.shape[0]

    def get_trial(self, index):
        """Return a single raw trial as an EpochsArray."""
        return mne.EpochsArray(
            self._eeg_data[index:index + 1],
            info=self._epochs.info,
            events=self._events[index:index + 1],
            tmin=self._epochs.tmin,
            verbose=False,
        )

    @staticmethod
    def _load_subject_epochs(subject_id):
        print("Loading data...")

        data_path = DATA_ROOT / "raw"
        subject_path = data_path / subject_id
        if not subject_path.exists():
            raise FileNotFoundError(f"Subject directory not found at {subject_path}")

        forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
        if not forward_path.exists():
            raise FileNotFoundError(
                f"Forward solution not found at {forward_path}. "
                f"Run: python {DATA_ROOT / 'build_fsaverage_forward.py'}"
            )
        channel_order = mne.read_forward_solution(forward_path).ch_names
        montage = mne.channels.make_standard_montage('standard_1005')

        epochs = mne.read_epochs_eeglab(
            os.path.join(subject_path, f'{subject_id}_task-tep_all_eeg.set')
        )
        epochs.pick(COMMON_CHANNELS)
        epochs.reorder_channels(channel_order)
        epochs.set_montage(None)
        epochs.set_montage(montage)

        if channel_order != epochs.ch_names:
            raise ValueError(f"Channel order mismatch: {channel_order} vs {epochs.ch_names}")

        return epochs
