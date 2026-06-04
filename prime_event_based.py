"""PRIME decider module for NeuroSimo.

Implements the PRIME online pipeline equivalent to simulate_online.py:
  1. Calibration phase (first N_CALIBRATION_TRIALS events):
     Accumulate raw trials, then batch-calibrate preprocessing, dipole fitting,
     TEP normalization, and predictor alignment.
  2. Intervention phase (remaining events):
     Preprocess each trial, fit dipole, compute label, predict, and finetune.

Each trial is delivered as a process_event() call (trial midpoints = events).
The EEG buffer around the event contains the full trial window.

Requires:
  - A pretrained PRIME model checkpoint (.pt)
  - A global back-rotation matrix (.npy)
  - An MNE forward solution (.fif) for dipole fitting

See simulate_online.py for the offline simulation equivalent.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from prime.online_predictor import OnlinePredictor
from prime.preprocessing.preprocessor import Preprocessor
from prime.preprocessing.dipole_fitter import DipoleFitter
from prime.tep_normalizer import TEPNormalizer

# ---------------------------------------------------------------------------
# Paths — adjust per setup
# ---------------------------------------------------------------------------

FORWARD_PATH = Path("data") / "fsaverage" / "fsaverage-fwd.fif"
PRETRAINED_MODEL_PATH = Path("offline_results") / "train" / "pretrained.pt"
GLOBAL_BACKROTATION_PATH = Path("offline_results") / "train" / "global_backrotation.npy"

# ---------------------------------------------------------------------------
# Protocol parameters
# ---------------------------------------------------------------------------

N_CALIBRATION_TRIALS = 125
SEED = 42

# Event sample window: must cover the full trial range needed by the Preprocessor.
# ICA needs [-1.1, -0.005], post needs [-0.03, 0.1]. Use the dataset range for safety.
EVENT_SAMPLE_WINDOW = [-1.3, 0.5998]


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

        # Trial counter
        self.trial_count = 0
        self.is_calibrated = False

        self.preprocessor = Preprocessor(FORWARD_PATH)
        self.dipole_fitter = DipoleFitter(FORWARD_PATH)
        self.normalizer = TEPNormalizer()

        global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)
        self.predictor = OnlinePredictor(
            global_backrotation,
            model_path=PRETRAINED_MODEL_PATH,
            seed=SEED,
        )

        print(f"PRIME decider ready  subject={subject_id}  fs={sampling_frequency}  "
              f"eeg={num_eeg_channels}  emg={num_emg_channels}")

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_configuration(self) -> dict[str, Any]:
        return {
            "event_sample_window": EVENT_SAMPLE_WINDOW,
            "sample_window": [-0.5, 0.0],
            "warm_up_rounds": 0,
        }

    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int, is_warm_up: bool) -> dict[str, Any] | None:
        """Process EEG/EMG buffer periodically."""
        pass

    # ==================================================================
    # Event processing (one call per trial)
    # ==================================================================

    def process_event(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        """Process a trial event. Mirrors simulate_online.py trial-by-trial logic."""

        if not self.is_calibrated:
            # --- Calibration phase ---
            self.preprocessor.add_trial(eeg_buffer, time_offsets)
            self.trial_count += 1
            print(f"Calibration trial {self.trial_count}/{N_CALIBRATION_TRIALS}")

            if self.trial_count >= N_CALIBRATION_TRIALS:
                self.run_calibration()

        else:
            # --- Intervention phase ---
            self.trial_count += 1
            processed_pre = self.preprocessor.preprocess_pre(eeg_buffer, time_offsets)
            processed_post = self.preprocessor.preprocess_post(eeg_buffer, time_offsets)

            if processed_pre is None or processed_post is None:
                print(f"Trial {self.trial_count}: REJECTED by preprocessing")
                return None

            amplitude = self.dipole_fitter.fit_trial(processed_post)
            label = self.normalizer.transform(amplitude)
            probability = self.predictor.predict(processed_pre)
            self.predictor.finetune(processed_pre, label)

            print(f"Trial {self.trial_count}: prediction={probability:.6f}  label={label:.6f}")

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
