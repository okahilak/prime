#!/usr/bin/env python3
"""
Convert online-preprocessing .npy buffer arrays to MNE Epochs (.fif).

Reads ``data/processed/{subject}/{subject}_{label}_{key}_buffer.npy`` files
written by ``prime.preprocessing.preprocess`` and saves sibling
``.fif`` files with the same basename.

Usage (from repo root):
  python -m prime.tools.npy_buffers_to_fif 21
  python -m prime.tools.npy_buffers_to_fif sub-021
"""

import argparse
import re
from pathlib import Path
from typing import Callable

import mne
import numpy as np

from prime.datasets import _channel_names_for_epochs
from prime.prime_config import (
    epoch_n_times,
    get_dipole_time_range,
    get_model_time_range,
    get_post_time_range,
    get_processed_sfreq,
    get_qc_time_range,
)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "offline_data"
PROCESSED_DIR = DATA_ROOT / "processed"

BUFFER_KEYS = ("qc", "model", "post", "dipole")

BUFFER_TIME_RANGE: dict[str, Callable[[], tuple[float, float]]] = {
    "qc": get_qc_time_range,
    "model": get_model_time_range,
    "post": get_post_time_range,
    "dipole": get_dipole_time_range,
}

BUFFER_FILE_PATTERN = re.compile(
    r"^(?P<subject>.+)_(?P<label>calibration|intervention)_(?P<key>"
    + "|".join(BUFFER_KEYS)
    + r")_buffer\.npy$"
)

mne.set_log_level("ERROR")


def normalize_subject_id(subject: str) -> str:
    """Map ``21`` or ``sub-21`` to ``sub-021``; leave other ids unchanged."""
    subject = subject.strip()
    match = re.fullmatch(r"sub-(\d+)(.*)", subject)
    if match:
        return f"sub-{int(match.group(1)):03d}{match.group(2)}"
    if subject.isdigit():
        return f"sub-{int(subject):03d}"
    return subject


def _epochs_from_buffer_npy(path: Path, key: str) -> mne.Epochs:
    data = np.load(path)
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D array (trials, channels, times), got shape {data.shape} at {path}"
        )

    tmin, tmax = BUFFER_TIME_RANGE[key]()
    sfreq = get_processed_sfreq()
    n_channels = len(_channel_names_for_epochs(data.shape[1]))
    if data.shape[1] != n_channels:
        raise ValueError(
            f"Channel dimension mismatch at {path}: shape[1]={data.shape[1]}, "
            f"expected {n_channels} channels"
        )
    expected_n_times = epoch_n_times(tmin, tmax, sfreq)
    if data.shape[2] != expected_n_times:
        raise ValueError(
            f"Time dimension mismatch at {path}: shape[2]={data.shape[2]}, "
            f"expected {expected_n_times} for [{tmin}, {tmax}] at {sfreq} Hz"
        )

    n_trials = data.shape[0]
    ch_names = _channel_names_for_epochs(data.shape[1])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    events = np.column_stack([
        np.arange(n_trials),
        np.zeros(n_trials, dtype=int),
        np.ones(n_trials, dtype=int),
    ])
    return mne.EpochsArray(data, info=info, events=events, tmin=tmin, verbose=False)


def convert_subject_buffers(subject: str) -> None:
    subject_id = normalize_subject_id(subject)
    subject_dir = PROCESSED_DIR / subject_id
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Processed subject directory not found: {subject_dir}")

    buffer_files = sorted(subject_dir.glob(f"{subject_id}_*_buffer.npy"))
    if not buffer_files:
        raise FileNotFoundError(
            f"No *_buffer.npy files found under {subject_dir}. "
            "Run preprocessing first."
        )

    converted = 0
    for npy_path in buffer_files:
        match = BUFFER_FILE_PATTERN.match(npy_path.name)
        if match is None:
            print(f"  SKIP (unrecognized name): {npy_path.name}")
            continue

        key = match.group("key")
        epochs = _epochs_from_buffer_npy(npy_path, key)
        fif_path = npy_path.with_suffix(".fif")
        epochs.save(fif_path, overwrite=True)
        print(f"  {npy_path.name} -> {fif_path.name} ({len(epochs)} epochs)")
        converted += 1

    if converted == 0:
        raise RuntimeError(f"No buffer files converted for {subject_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert preprocessing .npy buffers to MNE .fif epochs."
    )
    parser.add_argument(
        "subject",
        type=str,
        help="Subject id (e.g. 21, sub-021).",
    )
    args = parser.parse_args()

    subject_id = normalize_subject_id(args.subject)
    print(f"Converting buffers for {subject_id} under {PROCESSED_DIR}")
    convert_subject_buffers(args.subject)
    print("Done.")


if __name__ == "__main__":
    main()
