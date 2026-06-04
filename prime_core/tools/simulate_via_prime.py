"""Simulate the C++ backend feeding buffers through the event-based Decider.

Reads the CSV+JSON simulator dataset, extracts buffers around each event
(mimicking what the NeuroSimo backend does), and feeds them through
``simulate_event_based.Decider``. Intended to mirror ``prime_core.test_by_trial``
on the same short simulator dataset.

Usage (from repository root, with venv activated):
    python -m prime_core.tools.simulate_via_prime SUBJECT_ID [--n-trials N]

Examples:
    python -m prime_core.tools.simulate_via_prime 21 --n-trials 5    # First 5 calib trials (data check)
    python -m prime_core.tools.simulate_via_prime 21 --n-trials 130  # 125 calib + 5 intervention
"""

# Force single-threaded BLAS/LAPACK BEFORE importing numpy/scipy/torch.
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

from simulate_event_based import Decider, N_CALIBRATION_TRIALS
from prime_core.prime_config import get_calibration_time_range

DATA_ROOT = ROOT_DIR / "offline_data"


def default_short_dataset(subject_id: int) -> Path:
    subject_id_str = f"sub-{subject_id:03d}"
    return DATA_ROOT / "simulator" / subject_id_str / f"{subject_id_str}-short.json"


def load_dataset(json_path: Path):
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
    raw_data = np.loadtxt(dataset_dir / metadata["data_file"], delimiter=",")
    print(f"  Data shape: {raw_data.shape}")

    return raw_data, event_times, sfreq, n_eeg_channels, n_emg_channels


def extract_buffer(
    raw_data,
    event_sample,
    sfreq,
    window,
    n_eeg_channels,
    n_emg_channels,
):
    """Extract EEG/EMG buffers and time_offsets around an event."""
    start_offset = int(round(window[0] * sfreq))
    n_samples = int(round((window[1] - window[0]) * sfreq)) + 1

    start_sample = event_sample + start_offset
    end_sample = start_sample + n_samples

    eeg_buffer = raw_data[start_sample:end_sample, :n_eeg_channels]
    if n_emg_channels > 0:
        emg_buffer = raw_data[
            start_sample:end_sample,
            n_eeg_channels : n_eeg_channels + n_emg_channels,
        ]
    else:
        emg_buffer = np.zeros((n_samples, 0))

    time_offsets = np.arange(n_samples) / sfreq + window[0]
    return eeg_buffer, emg_buffer, time_offsets


def run_comparison(subject_id: int, n_trials: int, dataset_path: Path):
    event_sample_window = get_calibration_time_range()

    raw_data, event_times, sfreq, n_eeg_channels, n_emg_channels = load_dataset(
        dataset_path
    )
    event_samples = np.round(event_times * sfreq).astype(int)
    n_trials = min(n_trials, len(event_times))
    print(f"\nRunning {n_trials} trials (calibration={N_CALIBRATION_TRIALS})")
    print(f"Event sample window: {event_sample_window}")
    print(
        "Samples per trial:",
        int(round((event_sample_window[1] - event_sample_window[0]) * sfreq)) + 1,
    )

    print("\n--- Running DECIDER (simulate_event_based.process_event) ---")
    decider = Decider(
        subject_id=subject_id,
        num_eeg_channels=n_eeg_channels,
        num_emg_channels=n_emg_channels,
        sampling_frequency=int(sfreq),
    )

    for trial_idx in range(n_trials):
        eeg_buffer, emg_buffer, time_offsets = extract_buffer(
            raw_data,
            event_samples[trial_idx],
            sfreq,
            event_sample_window,
            n_eeg_channels,
            n_emg_channels,
        )

        decider.process_event(
            reference_time=event_times[trial_idx],
            reference_index=int(event_samples[trial_idx]),
            time_offsets=time_offsets,
            eeg_buffer=eeg_buffer,
            emg_buffer=emg_buffer,
            is_coil_at_target=True,
            stage_name="intervention",
            trial_in_stage=trial_idx,
        )

    print("\nDecider run complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Feed simulator CSV buffers through simulate_event_based.Decider"
    )
    parser.add_argument(
        "subject_id",
        type=int,
        help="Subject ID (e.g. 21 for sub-021)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=130,
        help="Total trials to process (default: 130 = 125 calib + 5 intervention)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to simulator dataset JSON metadata (default: sub-XXX-short.json)",
    )
    args = parser.parse_args()
    dataset_path = args.dataset or default_short_dataset(args.subject_id)
    run_comparison(args.subject_id, args.n_trials, dataset_path)


if __name__ == "__main__":
    main()
