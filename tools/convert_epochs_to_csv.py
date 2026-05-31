"""Convert epoched EEGLAB data to the EEG simulator CSV+JSON format.

This script loads epoched EEG data, picks and reorders channels to match
the forward solution, then reconstructs a continuous raw signal by placing
epochs at their original positions and filling gaps with reproducible random
noise. The output is a JSON metadata file, a CSV data file, and an event file.

Usage:
    python convert_epochs_to_csv.py <subject_id> [--output-dir <dir>]

Example:
    python convert_epochs_to_csv.py sub-021 --output-dir /home/user/prime-data/simulator
"""

import argparse
import json
import os
from pathlib import Path

import mne
import numpy as np

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

RANDOM_SEED = 42


def load_and_prepare_epochs(subject_id):
    """Load epochs from EEGLAB, pick and reorder channels."""
    data_path = DATA_ROOT / "raw" / subject_id
    if not data_path.exists():
        raise FileNotFoundError(f"Subject directory not found at {data_path}")

    forward_path = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
    if not forward_path.exists():
        raise FileNotFoundError(f"Forward solution not found at {forward_path}")

    channel_order = mne.read_forward_solution(forward_path).ch_names
    montage = mne.channels.make_standard_montage('standard_1005')

    epochs = mne.read_epochs_eeglab(
        os.path.join(data_path, f'{subject_id}_task-tep_all_eeg.set')
    )
    epochs.pick(COMMON_CHANNELS)
    epochs.reorder_channels(channel_order)
    epochs.set_montage(None)
    epochs.set_montage(montage)

    if channel_order != epochs.ch_names:
        raise ValueError(f"Channel order mismatch: {channel_order} vs {epochs.ch_names}")

    return epochs


def epochs_to_continuous(epochs):
    """Reconstruct continuous raw data from epochs, filling gaps with noise.

    Returns:
        raw_data: numpy array of shape (total_samples, n_channels)
        event_samples: 1D array of event sample indices in the continuous data
    """
    sfreq = epochs.info['sfreq']
    tmin = epochs.tmin
    data = epochs.get_data(copy=False)  # (n_epochs, n_channels, n_times)
    events = epochs.events  # (n_epochs, 3), column 0 is sample index

    n_epochs, n_channels, n_times = data.shape
    start_offset = int(round(tmin * sfreq))  # negative

    # Determine total length of continuous signal
    # Each epoch spans: event_sample + start_offset to event_sample + start_offset + n_times - 1
    last_event_sample = events[-1, 0]
    total_samples = last_event_sample + start_offset + n_times

    # Add a small buffer at the end (1 second)
    total_samples += int(sfreq)

    print(f"Reconstructing continuous signal: {total_samples} samples "
          f"({total_samples / sfreq:.1f} seconds), {n_channels} channels")

    # Fill with reproducible random noise
    rng = np.random.default_rng(RANDOM_SEED)
    # Use noise with similar scale to the data
    data_std = np.std(data)
    raw_data = rng.normal(0, data_std * 0.1, size=(total_samples, n_channels))

    # Place epochs at their correct positions
    for i in range(n_epochs):
        event_sample = events[i, 0]
        epoch_start = event_sample + start_offset
        epoch_end = epoch_start + n_times

        if epoch_start < 0:
            # Trim the beginning if it goes before sample 0
            trim = -epoch_start
            raw_data[0:epoch_end, :] = data[i, :, trim:].T
        elif epoch_end > total_samples:
            # Trim the end if it goes past the total length
            trim = epoch_end - total_samples
            raw_data[epoch_start:total_samples, :] = data[i, :, :n_times - trim].T
        else:
            raw_data[epoch_start:epoch_end, :] = data[i, :, :].T

    event_samples = events[:, 0]
    return raw_data, event_samples


def write_dataset(output_dir, subject_id, raw_data, event_samples, sfreq, n_channels):
    """Write the CSV data, event file, and JSON metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_filename = f"{subject_id}_data.csv"
    event_filename = f"{subject_id}_events.csv"
    json_filename = f"{subject_id}.json"

    # Write CSV data (no header, comma-separated)
    # Use full double precision (17 significant digits) to avoid ICA divergence
    # from tiny numerical differences
    print(f"Writing data CSV ({raw_data.shape[0]} rows x {raw_data.shape[1]} cols)...")
    np.savetxt(
        output_dir / data_filename,
        raw_data,
        delimiter=',',
        # TODO: Would %.9g be sufficient?
        fmt='%.17g',
    )

    # Write event file (event times in seconds, one per line)
    event_times = event_samples / sfreq
    print(f"Writing event file ({len(event_times)} events)...")
    np.savetxt(
        output_dir / event_filename,
        event_times,
        fmt='%.7f',
    )

    # Write JSON metadata
    metadata = {
        "name": f"{subject_id} EEG dataset",
        "data_file": data_filename,
        "session": {
            "sampling_frequency": int(sfreq),
            "num_eeg_channels": n_channels,
            "num_emg_channels": 0,
        },
        "loop": False,
        "event_file": event_filename,
    }

    with open(output_dir / json_filename, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset written to {output_dir}")
    print(f"  Metadata: {json_filename}")
    print(f"  Data: {data_filename}")
    print(f"  Events: {event_filename}")

    return output_dir / json_filename


def main():
    parser = argparse.ArgumentParser(
        description="Convert epoched EEGLAB data to EEG simulator CSV+JSON format."
    )
    parser.add_argument("subject_id", help="Subject ID (e.g., sub-021)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: ~/prime-data/simulator/<subject_id>)",
    )
    parser.add_argument(
        "--short",
        action="store_true",
        help="Only use the first 200 trials; output files are named <subject_id>-short_*",
    )
    args = parser.parse_args()

    mne.set_log_level("WARNING")

    if args.output_dir is None:
        output_dir = DATA_ROOT / "simulator" / args.subject_id
    else:
        output_dir = Path(args.output_dir)

    print(f"Loading epochs for {args.subject_id}...")
    epochs = load_and_prepare_epochs(args.subject_id)

    output_subject_id = args.subject_id
    if args.short:
        epochs = epochs[:200]
        output_subject_id = f"{args.subject_id}-short"
        print(f"--short: using first {len(epochs)} epochs, output prefix: {output_subject_id}")

    print(f"Epochs: {epochs.get_data().shape}")
    print(f"  sfreq={epochs.info['sfreq']}, tmin={epochs.tmin}, tmax={epochs.tmax}")

    raw_data, event_samples = epochs_to_continuous(epochs)

    json_path = write_dataset(
        output_dir,
        output_subject_id,
        raw_data,
        event_samples,
        epochs.info['sfreq'],
        len(epochs.ch_names),
    )

    print(f"\nDone! JSON metadata at: {json_path}")


if __name__ == "__main__":
    main()
