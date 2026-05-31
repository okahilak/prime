"""Provides raw EEG trials by index from a CSV+JSON simulator dataset."""

import json
from pathlib import Path

import mne
import numpy as np

DATA_ROOT = Path("~/prime-data").expanduser()

CHANNEL_NAMES = [
    'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
    'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
    'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
    'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz',
    'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
    'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz',
    'T7', 'T8', 'TP7', 'TP8',
]

# Must match what was used to create the dataset
TMIN = -1.3
TMAX = 0.5998

class TrialLoaderFromCsv:
    """Loads all epochs from a CSV+JSON simulator dataset and provides raw trials by index."""

    def __init__(self, json_path):
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"JSON metadata not found at {json_path}")

        with open(json_path) as f:
            metadata = json.load(f)

        dataset_dir = json_path.parent
        sfreq = metadata["session"]["sampling_frequency"]
        n_eeg_channels = metadata["session"]["num_eeg_channels"]

        # Load event times
        event_file = metadata.get("event_file")
        if event_file is None:
            raise ValueError("No event_file specified in metadata")
        event_times = np.loadtxt(dataset_dir / event_file)
        event_samples = np.round(event_times * sfreq).astype(int)

        # Load raw continuous data
        print("Loading CSV data...")
        raw_data = np.loadtxt(dataset_dir / metadata["data_file"], delimiter=',')
        # raw_data shape: (total_samples, n_channels)

        n_channels = raw_data.shape[1]
        if n_channels < n_eeg_channels:
            raise ValueError(
                f"Data has {n_channels} columns but expected at least {n_eeg_channels} EEG channels"
            )

        # Extract only EEG channels
        eeg_data = raw_data[:, :n_eeg_channels]

        # Epoch the continuous data
        start_offset = int(round(TMIN * sfreq))

        # Calculate n_times from tmin, tmax, and the epoch length used during creation
        n_times = int(round((TMAX - TMIN) * sfreq)) + 1

        n_epochs = len(event_samples)
        epochs_data = np.zeros((n_epochs, n_eeg_channels, n_times))

        for i, ev_sample in enumerate(event_samples):
            epoch_start = ev_sample + start_offset
            epoch_end = epoch_start + n_times
            if epoch_start < 0 or epoch_end > eeg_data.shape[0]:
                raise ValueError(
                    f"Epoch {i} out of bounds: samples {epoch_start}:{epoch_end}, "
                    f"data length {eeg_data.shape[0]}"
                )
            epochs_data[i] = eeg_data[epoch_start:epoch_end].T

        # Build MNE info and epochs
        montage = mne.channels.make_standard_montage('standard_1005')
        info = mne.create_info(
            ch_names=CHANNEL_NAMES[:n_eeg_channels],
            sfreq=sfreq,
            ch_types='eeg',
        )
        info.set_montage(montage)

        events = np.column_stack([
            event_samples,
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ])

        self._epochs = mne.EpochsArray(
            epochs_data,
            info=info,
            events=events,
            tmin=TMIN,
            verbose=False,
        )
        self._eeg_data = self._epochs.get_data(copy=False)
        self._events = self._epochs.events

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
