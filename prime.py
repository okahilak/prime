"""PRIME decider module for NeuroSimo.

Implements protocols/prime.yaml (PRIME-TEP validation):

  1. Calibration stage (100 predetermined high-ITI trials):
     Accumulate trials via process_pulse, then batch-calibrate in calibrate_prime task.
  2. Intervention blocks (75% PRIME-guided periodic, 25% predetermined high-ITI):
     process_periodic: rolling-window QC + prediction; schedule pulse when excitable.
     process_pulse: post-stimulus preprocessing, dipole fit, label, finetune.
  3. Evaluation stages (predetermined low-ITI):
     Timed brain-state-independent pulses only (no finetuning).

Requires:
  - A pretrained PRIME model checkpoint (.pt)
  - A global back-rotation matrix (.npy)
  - An MNE forward solution (.fif) for dipole fitting
"""

import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np

from prime_core.online_predictor import OnlinePredictor
from prime_core.preprocessing.preprocessor import Preprocessor, crop_eeg_buffer
from prime_core.preprocessing.dipole_fitter import DipoleFitter
from prime_core.prime_config import (
    get_calibration_time_range,
    get_qc_time_range,
    get_post_initial_time_range,
)
from prime_core.tep_normalizer import TEPNormalizer

# ---------------------------------------------------------------------------
# Paths — adjust per setup
# ---------------------------------------------------------------------------

FORWARD_PATH = Path("offline_data") / "fsaverage" / "fsaverage-fwd.fif"
PRETRAINED_MODEL_PATH = Path("offline_results") / "train" / "pretrained.pt"
GLOBAL_BACKROTATION_PATH = Path("offline_results") / "train" / "global_backrotation.npy"

# ---------------------------------------------------------------------------
# Protocol parameters (protocols/prime.yaml, PRIME.md)
# ---------------------------------------------------------------------------

PREDICTION_THRESHOLD = 0.5
TRIGGER_OFFSET = 0.01

HIGH_ITI_MIN = 4.0
HIGH_ITI_MAX = 9.0
LOW_ITI_MIN = 2.5
LOW_ITI_MAX = 3.5

SEED = 42


def timed_ms(fn, /, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


class Decider:
    def __init__(
        self,
        subject_id: int,
        num_eeg_channels: int,
        num_emg_channels: int,
        sampling_frequency: int,
    ):
        self.subject_id = subject_id
        self.num_eeg_channels = num_eeg_channels
        self.num_emg_channels = num_emg_channels
        self.sampling_frequency = sampling_frequency

        self.calibration_tmin, self.calibration_tmax = get_calibration_time_range()

        # Quality control window
        self.qc_tmin, self.qc_tmax = get_qc_time_range()
        self.qc_window_size = self.qc_tmax - self.qc_tmin

        self.post_initial_tmin, self.post_initial_tmax = get_post_initial_time_range()

        self.rng = np.random.default_rng(SEED + subject_id)

        self.is_calibrated = False
        self.pending_pre: Optional[np.ndarray] = None

        global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)
        self.predictor = OnlinePredictor(global_backrotation, model_path=PRETRAINED_MODEL_PATH, seed=SEED)

        self.preprocessor = Preprocessor(FORWARD_PATH)
        self.dipole_fitter = DipoleFitter(FORWARD_PATH)
        self.normalizer = TEPNormalizer()

        print(
            f"PRIME decider ready  subject={subject_id}  fs={sampling_frequency}  "
            f"eeg={num_eeg_channels}  emg={num_emg_channels}"
        )

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_configuration(self) -> dict[str, Any]:
        return {
            "periodic_processing_interval": 0.01,
            "sample_window": [-self.qc_window_size, 0.0],
            "pulse_sample_window": [self.calibration_tmin, self.calibration_tmax],
            "warm_up_rounds": 0,
        }

    # ==================================================================
    # Protocol helpers
    # ==================================================================
    def is_intervention_stage(stage_name):
        return stage_name.startswith("intervention_block_")

    def is_evaluation_stage(stage_name):
        return stage_name.startswith("evaluation_")

    # ==================================================================
    # Predetermined trial timing
    # ==================================================================

    def process_predetermined(
            self, reference_time: float, stage_name: str, trial: int,
            trial_type: str) -> dict[str, Any] | None:
        if trial_type == "high_iti":
            iti = self.rng.uniform(HIGH_ITI_MIN, HIGH_ITI_MAX)
        elif trial_type == "low_iti":
            iti = self.rng.uniform(LOW_ITI_MIN, LOW_ITI_MAX)
        else:
            raise ValueError(f"Unknown predetermined trial type: {trial_type!r}")

        return {"trigger_offset": iti}

    # ==================================================================
    # Calibration task
    # ==================================================================

    def process_task(self, task_name: str) -> None:
        if task_name != "calibrate_prime":
            raise ValueError(f"Unknown task: {task_name!r}")
        self.run_calibration()

    # ==================================================================
    # Periodic processing (PRIME-guided intervention)
    # ==================================================================

    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int,
            is_warm_up: bool) -> dict[str, Any] | None:

        if is_warm_up or not self.is_calibrated:
            return None
        if not self.is_intervention_stage(stage_name):
            return None
        if self.pending_pre is not None:
            return None

        pre = self.preprocessor.preprocess_pre(eeg_buffer, time_offsets, online=True)
        if pre is None:
            return None

        probability, predict_ms = timed_ms(self.predictor.predict, pre)
        print(
            f"Rolling check t={reference_time:.3f}s: prediction={probability:.6f}  "
            f"prediction_time={predict_ms:.1f}ms"
        )
        if probability < PREDICTION_THRESHOLD:
            return None

        self.pending_pre = pre
        print(
            f"PRIME trigger at t={reference_time + TRIGGER_OFFSET:.3f}s  "
            f"prediction={probability:.6f}  prediction_time={predict_ms:.1f}ms"
        )
        return {"trigger_offset": TRIGGER_OFFSET}

    # ==================================================================
    # Pulse processing
    # ==================================================================

    def process_pulse(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:

        if stage_name == "calibration":
            self.preprocessor.add_trial(eeg_buffer, time_offsets)
            print(f"Calibration trial collected ({trial_in_stage + 1} valid in stage)")
            return None

        if self.is_evaluation_stage(stage_name):
            return None

        if not self.is_calibrated or not self.is_intervention_stage(stage_name):
            return None

        pre = self.pending_pre
        self.pending_pre = None

        post_buffer, post_time_offsets = crop_eeg_buffer(
            eeg_buffer,
            time_offsets,
            self.post_initial_tmin,
            self.post_initial_tmax,
        )
        post = self.preprocessor.preprocess_post(post_buffer, post_time_offsets)

        if pre is None:
            print(f"Trial {trial_in_stage + 1} ({stage_name}): REJECTED (pre-stimulus)")
            return {"trial_invalid": True}
        if post is None:
            print(f"Trial {trial_in_stage + 1} ({stage_name}): REJECTED (post-stimulus)")
            return {"trial_invalid": True}

        amplitude = self.dipole_fitter.fit_trial(post)
        label = self.normalizer.transform(amplitude)
        self.predictor.finetune(pre, label)
        print(f"Trial {trial_in_stage + 1} ({stage_name}): label={label:.6f}")
        return None

    # ==================================================================
    # Calibration
    # ==================================================================

    def run_calibration(self) -> None:
        print("Running calibration...")

        t0 = time.perf_counter()

        model_buffers, dipole_buffers = self.preprocessor.calibrate()
        amplitudes = self.dipole_fitter.calibrate(dipole_buffers)
        labels = self.normalizer.calibrate(amplitudes)
        self.predictor.calibrate(model_buffers, labels)
        self.predictor.warm_up()

        print(f"Calibration took {time.perf_counter() - t0:.2f} seconds")
        self.is_calibrated = True
