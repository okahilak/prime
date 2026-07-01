"""PRIME evaluation decider for NeuroSimo.

Implements protocols/prime_evaluation.yaml (post-intervention evaluation):

  Evaluation stage (100 predetermined single pulses):
    Timed, brain-state-independent single pulses (scheduled ITI). No prediction
    or finetuning — the raw pre/post EEG buffers are saved for each trial so the
    evaluation TEP can be computed offline.

The operator starts this protocol manually at each post-intervention time point
(0, 15, 30, 60 min). Each run detects how many evaluation runs already exist for
the subject/session and writes to its own trials_evaluation_<N>.csv, so repeated
runs never overwrite one another.
"""

import csv
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from prime_core.preprocessing.preprocessor import crop_eeg_buffer
from prime_core.prime_config import (
    get_calibration_time_range,
    get_qc_time_range,
    get_post_initial_time_range,
)
from util.magventure_tms import MagVentureTMS
from util.magventure_tms_mock import MockTMS

# ---------------------------------------------------------------------------
# Protocol parameters
# ---------------------------------------------------------------------------

ITI_MIN = 2.5
ITI_MAX = 5.5

SEED = 101

BUFFER_TOLERANCE = 0.005

EVALUATION_CSV_PATTERN = re.compile(r"trials_evaluation_(\d+)\.csv$")


class Decider:
    def __init__(self, subject_id: int, num_eeg_channels: int, num_emg_channels: int, sampling_frequency: int,
                 runtime_params: dict[str, Any]):
        self.subject_id = subject_id
        self.num_eeg_channels = num_eeg_channels
        self.num_emg_channels = num_emg_channels
        self.sampling_frequency = sampling_frequency

        self.mock_tms_device = runtime_params.get("mock_tms_device")
        self.is_open_loop_session = runtime_params["is_open_loop_session"]
        self.single_pulse_intensity = runtime_params["single_pulse_intensity"]
        self.tbs_intensity = runtime_params["tbs_intensity"]

        self.calibration_tmin, self.calibration_tmax = get_calibration_time_range()

        # Quality control window (used only to size the pre-stimulus sample window).
        self.qc_tmin, self.qc_tmax = get_qc_time_range()
        self.qc_window_size = self.qc_tmax - self.qc_tmin

        self.post_initial_tmin, self.post_initial_tmax = get_post_initial_time_range()

        self.rng = np.random.default_rng(SEED + subject_id)

        self.tms = MagVentureTMS() if not self.mock_tms_device else MockTMS()

        # Create results directory and this run's trials CSV file. The run index
        # is derived from the evaluation CSVs already present, so consecutive
        # manual runs land in trials_evaluation_1.csv, _2.csv, ...
        self.results_dir = Path("results") / str(subject_id) / ("open_loop" if self.is_open_loop_session else "prime")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.run_index = self.next_run_index()
        self.trials_csv = self.results_dir / f"trials_evaluation_{self.run_index}.csv"
        self.csv_fields = [
            "stage", "run_index", "trial_in_stage", "iti",
            "trial_start_time", "target_time", "pulse_time",
        ]
        with open(self.trials_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writeheader()
        self.current_trial: dict = {}

        print(
            f"PRIME evaluation decider ready  subject={subject_id}  run={self.run_index}  "
            f"fs={sampling_frequency}  eeg={num_eeg_channels}  emg={num_emg_channels}"
        )

    def next_run_index(self) -> int:
        """Next evaluation run index, based on existing trials_evaluation_<N>.csv files."""
        indices = [
            int(match.group(1))
            for path in self.results_dir.glob("trials_evaluation_*.csv")
            if (match := EVALUATION_CSV_PATTERN.search(path.name))
        ]
        return max(indices, default=0) + 1

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_configuration(self) -> dict[str, Any]:
        return {
            "sample_window": [-self.qc_window_size - BUFFER_TOLERANCE, 0.0],
            "pulse_sample_window": [self.calibration_tmin - BUFFER_TOLERANCE, self.calibration_tmax + BUFFER_TOLERANCE],
            "warm_up_rounds": 0,
        }

    # ==================================================================
    # Trial preparation (timing + stimulator arming)
    # ==================================================================

    def prepare_trial(self, start_time: float, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        """Arm the stimulator and schedule the predetermined trigger."""
        if stage_name != "evaluation":
            raise ValueError("Incorrect protocol, must be 3_evaluation.yaml for evaluation")

        iti = self.rng.uniform(ITI_MIN, ITI_MAX)

        self.current_trial = {
            "stage": stage_name,
            "run_index": self.run_index,
            "trial_in_stage": trial_in_stage,
            "iti": iti,
            "trial_start_time": start_time,
            "target_time": start_time + iti,
            "pulse_time": None,
        }

        self.tms.set_single_pulse(self.single_pulse_intensity)
        return {"trigger_offset": iti}

    # ==================================================================
    # Pulse processing
    # ==================================================================

    def process_pulse(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:

        print(f"Evaluation run {self.run_index} trial {trial_in_stage + 1} finished")

        self.current_trial["pulse_time"] = reference_time
        self.write_trial_row()
        self.save_raw_buffers(
            stage_name, trial_in_stage,
            self.extract_raw_pre_from_pulse(time_offsets, eeg_buffer)[0],
            self.extract_raw_post_from_pulse(time_offsets, eeg_buffer),
        )

    # ==================================================================
    # Trial logging
    # ==================================================================

    def write_trial_row(self) -> None:
        row = {field: self.current_trial.get(field) for field in self.csv_fields}
        with open(self.trials_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writerow(row)

    # ==================================================================
    # Raw buffer extraction and saving
    # ==================================================================

    def extract_raw_pre_from_pulse(
            self, time_offsets: np.ndarray, eeg_buffer: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return crop_eeg_buffer(eeg_buffer, time_offsets, self.qc_tmin, self.qc_tmax)

    def extract_raw_post_from_pulse(
            self, time_offsets: np.ndarray, eeg_buffer: np.ndarray
    ) -> np.ndarray:
        raw_post, _ = crop_eeg_buffer(
            eeg_buffer, time_offsets, self.post_initial_tmin, self.post_initial_tmax
        )
        return raw_post

    def save_raw_buffers(
            self, stage_name: str, trial_in_stage: int,
            raw_pre: np.ndarray, raw_post: np.ndarray
    ) -> None:
        # Include the run index so repeated evaluation runs don't overwrite buffers.
        stem = f"{stage_name}_{self.run_index}_{trial_in_stage:04d}"
        np.save(self.results_dir / f"{stem}_pre_raw.npy", raw_pre)
        np.save(self.results_dir / f"{stem}_post_raw.npy", raw_post)
