"""PRIME baseline decider for NeuroSimo.

Implements protocols/prime_baseline.yaml (baseline TEP assessment):

  Baseline stage (100 predetermined single pulses):
    Timed, brain-state-independent single pulses (scheduled ITI). No calibration,
    prediction, or finetuning — the raw pre/post EEG buffers are saved for each
    trial so the baseline TEP can be computed offline.

Writes trial metadata to results/<subject>/<session>/trials_baseline.csv.
"""

import csv
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

SEED = 42

BUFFER_TOLERANCE = 0.005


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
        self.overwrite_existing_results = runtime_params.get("overwrite_existing_results", False)

        self.calibration_tmin, self.calibration_tmax = get_calibration_time_range()

        # Quality control window (used only to size the pre-stimulus sample window).
        self.qc_tmin, self.qc_tmax = get_qc_time_range()
        self.qc_window_size = self.qc_tmax - self.qc_tmin

        self.post_initial_tmin, self.post_initial_tmax = get_post_initial_time_range()

        self.rng = np.random.default_rng(SEED + subject_id)

        self.tms = MagVentureTMS() if not self.mock_tms_device else MockTMS()

        # Create results directory and trials CSV file.
        self.results_dir = Path("results") / str(subject_id) / ("open_loop" if self.is_open_loop_session else "prime")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trials_csv = self.results_dir / "trials_baseline.csv"
        if self.trials_csv.exists() and not self.overwrite_existing_results:
            raise FileExistsError(
                f"Baseline results already exist: {self.trials_csv} — "
                "baseline may have already been run for this subject and session type. "
                "Delete the file or enable overwriting."
            )
        self.csv_fields = [
            "stage", "trial_in_stage", "iti",
            "trial_start_time", "target_time", "pulse_time",
        ]
        with open(self.trials_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writeheader()
        self.current_trial: dict = {}

        print(
            f"PRIME baseline decider ready  subject={subject_id}  fs={sampling_frequency}  "
            f"eeg={num_eeg_channels}  emg={num_emg_channels}"
        )

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_configuration(self) -> dict[str, Any]:
        return {
            "periodic_processing_interval": 0.01,
            "sample_window": [-self.qc_window_size - BUFFER_TOLERANCE, 0.0],
            "pulse_sample_window": [self.calibration_tmin - BUFFER_TOLERANCE, self.calibration_tmax + BUFFER_TOLERANCE],
            "warm_up_rounds": 0,
        }

    # ==================================================================
    # Trial preparation (timing + stimulator arming)
    # ==================================================================

    def prepare_trial(self, start_time: float, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        """Arm the stimulator and schedule the predetermined trigger."""
        if stage_name != "baseline":
            raise ValueError(f"Unknown stage: {stage_name!r}")

        iti = self.rng.uniform(ITI_MIN, ITI_MAX)

        self.current_trial = {
            "stage": stage_name,
            "trial_in_stage": trial_in_stage,
            "iti": iti,
            "trial_start_time": start_time,
            "target_time": start_time + iti,
            "pulse_time": None,
        }

        self.tms.set_single_pulse(self.single_pulse_intensity)
        return {"trigger_offset": iti}

    # ==================================================================
    # Periodic processing (unused: baseline triggers are all predetermined)
    # ==================================================================

    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int,
            is_warm_up: bool) -> dict[str, Any] | None:
        return None

    # ==================================================================
    # Pulse processing
    # ==================================================================

    def process_pulse(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        if stage_name != "baseline":
            raise ValueError(f"Unknown stage: {stage_name!r}")

        print(f"Baseline trial {trial_in_stage + 1} finished")

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
        stem = f"{stage_name}_{trial_in_stage:04d}"
        np.save(self.results_dir / f"{stem}_pre_raw.npy", raw_pre)
        np.save(self.results_dir / f"{stem}_post_raw.npy", raw_post)
