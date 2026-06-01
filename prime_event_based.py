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

import hashlib
import time
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import mne
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
mne.set_log_level("ERROR")

# Add the prime/ subdirectory to path for imports
PRIME_DIR = Path(__file__).parent / "prime"
sys.path.insert(0, str(PRIME_DIR))
sys.path.insert(0, str(PRIME_DIR / "online_preprocessing"))

from online_predictor import OnlinePredictor
from online_preprocessing.preprocessor import Preprocessor, crop_eeg_buffer
from online_preprocessing.dipole_fitter import DipoleFitter
from prime_config import get_raw_post_epoch_time_range, get_raw_pre_epoch_time_range
from tep_normalizer import TEPNormalizer

# ---------------------------------------------------------------------------
# Paths — adjust per setup
# ---------------------------------------------------------------------------

PRIME_DIR = Path(__file__).parent / "prime"
DATA_ROOT = Path(__file__).parent / "data"

FORWARD_PATH = DATA_ROOT / "fsaverage" / "fsaverage-fwd.fif"
PRETRAINED_MODEL_PATH = PRIME_DIR / "results" / "train" / "pretrained.pt"
GLOBAL_BACKROTATION_PATH = PRIME_DIR / "results" / "train" / "global_backrotation.npy"

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

        self._raw_pre_tmin, self._raw_pre_tmax = get_raw_pre_epoch_time_range()
        self._raw_post_tmin, self._raw_post_tmax = get_raw_post_epoch_time_range()

        self.preprocessor = Preprocessor(str(FORWARD_PATH))
        self.dipole_fitter = DipoleFitter(str(FORWARD_PATH))
        self.normalizer = TEPNormalizer()

        global_backrotation = np.load(str(GLOBAL_BACKROTATION_PATH))
        self.predictor = OnlinePredictor(
            global_backrotation,
            model_path=str(PRETRAINED_MODEL_PATH),
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
            "warm_up_rounds": 3,
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

        raw_pre = crop_eeg_buffer(
            eeg_buffer, time_offsets, self._raw_pre_tmin, self._raw_pre_tmax,
        )
        raw_post = crop_eeg_buffer(
            eeg_buffer, time_offsets, self._raw_post_tmin, self._raw_post_tmax,
        )

        pre_checksum = hashlib.sha256(raw_pre.tobytes()).hexdigest()
        print(f"Pre-epoch sha256={pre_checksum}")
        post_checksum = hashlib.sha256(raw_post.tobytes()).hexdigest()
        print(f"Post-epoch sha256={post_checksum}")


        if not self.is_calibrated:
            # --- Calibration phase ---
            self.preprocessor.add_raw_pre_epoch(raw_pre)
            self.preprocessor.add_raw_post_epoch(raw_post)
            self.trial_count += 1
            print(f"Calibration trial {self.trial_count}/{N_CALIBRATION_TRIALS}")

            if self.trial_count >= N_CALIBRATION_TRIALS:
                self._run_calibration()

        else:
            # --- Intervention phase ---
            self.trial_count += 1
            processed_pre = self.preprocessor.preprocess_pre(raw_pre)
            processed_post = self.preprocessor.preprocess_post(raw_post)

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

    def _run_calibration(self) -> None:
        """Run the full calibration pipeline (matches simulate_online.py)."""
        print("\n" + "=" * 60)
        print("RUNNING CALIBRATION")
        print("=" * 60)

        cal_pre, cal_post = self.preprocessor.calibrate()
        print(f"  Preprocessor done: {len(cal_pre)} trials survived rejection")

        amplitudes = self.dipole_fitter.calibrate(cal_post)
        print(f"  Dipole fitter calibrated")

        labels = self.normalizer.calibrate(amplitudes)
        print(f"  TEP normalizer calibrated")

        self.predictor.calibrate(cal_pre, labels)
        print(f"  Predictor calibrated")

        self.is_calibrated = True
        print("=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60 + "\n")

