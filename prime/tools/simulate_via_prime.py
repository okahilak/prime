"""Simulate the C++ backend feeding buffers through prime.py's process_event.

This script reads the CSV+JSON simulator dataset, extracts buffers around each
event (mimicking what the C++ NeuroSimo backend does), and feeds them through
the Decider class in prime.py. Results are compared against simulate_online.py
run on the same data.

Usage (from workspace root, with venv activated):
    python simulation/simulate_via_prime.py [--n-trials N]

Examples:
    python simulation/simulate_via_prime.py --n-trials 5    # First 5 calib trials (data check)
    python simulation/simulate_via_prime.py --n-trials 130  # 125 calib + 5 intervention
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add paths for imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "prime"))
sys.path.insert(0, str(ROOT_DIR / "prime" / "online_preprocessing"))

from prime import Decider, EVENT_SAMPLE_WINDOW

# Same constants as simulate_online.py
N_CALIBRATION_TRIALS = 125
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
SEED = 42


def load_dataset(json_path):
    """Load continuous data and event times from the simulator dataset."""
    json_path = Path(json_path)
    with open(json_path) as f:
        metadata = json.load(f)

    dataset_dir = json_path.parent
    sfreq = metadata["session"]["sampling_frequency"]
    n_eeg_channels = metadata["session"]["num_eeg_channels"]
    n_emg_channels = metadata["session"]["num_emg_channels"]

    event_times = np.loadtxt(dataset_dir / metadata["event_file"])

    print(f"Loading CSV data from {metadata['data_file']}...")
    raw_data = np.loadtxt(dataset_dir / metadata["data_file"], delimiter=',')
    print(f"  Data shape: {raw_data.shape}")

    return raw_data, event_times, sfreq, n_eeg_channels, n_emg_channels


def extract_buffer(raw_data, event_sample, sfreq, window, n_eeg_channels, n_emg_channels):
    """Extract EEG/EMG buffers and time_offsets around an event.

    Mimics what the C++ backend does: provides samples covering the window
    [window[0], window[1]] around the event at the given sampling frequency.

    The number of samples is: int(round((window[1] - window[0]) * sfreq)) + 1
    (same formula as TrialLoaderFromCsv uses for n_times).
    """
    start_offset = int(round(window[0] * sfreq))
    n_samples = int(round((window[1] - window[0]) * sfreq)) + 1

    start_sample = event_sample + start_offset
    end_sample = start_sample + n_samples

    eeg_buffer = raw_data[start_sample:end_sample, :n_eeg_channels]
    if n_emg_channels > 0:
        emg_buffer = raw_data[start_sample:end_sample, n_eeg_channels:n_eeg_channels + n_emg_channels]
    else:
        emg_buffer = np.zeros((n_samples, 0))

    # Time of each sample relative to the event
    time_offsets = np.arange(n_samples) / sfreq + window[0]

    return eeg_buffer, emg_buffer, time_offsets


def run_comparison(n_trials):
    """Run the Decider path for n_trials."""

    dataset_path = DATA_ROOT / "simulator" / "sub-021" / "sub-021-short.json"

    # --- Load data ---
    raw_data, event_times, sfreq, n_eeg_channels, n_emg_channels = load_dataset(dataset_path)
    event_samples = np.round(event_times * sfreq).astype(int)
    n_trials = min(n_trials, len(event_times))
    print(f"\nRunning {n_trials} trials (calibration={N_CALIBRATION_TRIALS})")
    print(f"Event sample window: {EVENT_SAMPLE_WINDOW}")
    print(f"Samples per trial: {int(round((EVENT_SAMPLE_WINDOW[1] - EVENT_SAMPLE_WINDOW[0]) * sfreq)) + 1}")

    # --- Run Decider pipeline ---
    print("\n--- Running DECIDER (prime.py process_event) ---")
    decider = Decider(
        subject_id="sub-021-short",
        num_eeg_channels=n_eeg_channels,
        num_emg_channels=n_emg_channels,
        sampling_frequency=int(sfreq),
    )

    for trial_idx in range(n_trials):
        eeg_buffer, emg_buffer, time_offsets = extract_buffer(
            raw_data, event_samples[trial_idx], sfreq,
            EVENT_SAMPLE_WINDOW, n_eeg_channels, n_emg_channels,
        )

        result = decider.process_event(
            reference_time=event_times[trial_idx],
            reference_index=int(event_samples[trial_idx]),
            time_offsets=time_offsets,
            eeg_buffer=eeg_buffer,
            emg_buffer=emg_buffer,
            is_coil_at_target=True,
            stage_name="intervention",
            trial_in_stage=trial_idx,
        )

        if result is not None:
            print(f"  Dec trial {trial_idx}: pred={result['prediction']:.6f}  label={result['label']:.6f}")

    print("\nDecider run complete.")


def main():
    parser = argparse.ArgumentParser(description="Compare Decider vs simulate_online.py")
    parser.add_argument("--n-trials", type=int, default=130,
                        help="Total trials to process (default: 130 = 125 calib + 5 intervention)")
    args = parser.parse_args()

    run_comparison(args.n_trials)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
