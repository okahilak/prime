"""PRIME decider module for NeuroSimo.

Implements the PRIME online pipeline equivalent to simulate_online.py:
  1. Calibration phase (first N_CALIBRATION_TRIALS events):
     Accumulate raw pre epochs via process_periodic (5 ms before each event)
     and raw post epochs via process_event, then batch-calibrate.
  2. Intervention phase (remaining events):
     process_periodic: preprocess pre + predict (5 ms before pulse).
     process_event: preprocess post, fit dipole, label, finetune using stored pre.

Requires:
  - A pretrained PRIME model checkpoint (.pt)
  - A global back-rotation matrix (.npy)
  - An MNE forward solution (.fif) for dipole fitting

See simulate_online.py for the offline simulation equivalent.
"""

import hashlib
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

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
from online_preprocessing.preprocessor import Preprocessor
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

# process_periodic runs this many seconds before each TMS event (pre-stim window end).
EVENT_LOOKAHEAD_SEC = 0.005


@contextmanager
def _profile(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[profile] {label}: {elapsed * 1000:.1f}ms")


class Decider:
    def __init__(
        self,
        subject_id: str,
        num_eeg_channels: int,
        num_emg_channels: int,
        sampling_frequency: int,
    ):
        self.subject_id = subject_id
        self.num_eeg_channels = num_eeg_channels
        self.num_emg_channels = num_emg_channels
        self.sampling_frequency = sampling_frequency

        self.trial_count = 0
        self.is_calibrated = False

        self._raw_pre_tmin, self._raw_pre_tmax = get_raw_pre_epoch_time_range()
        self._raw_post_tmin, self._raw_post_tmax = get_raw_post_epoch_time_range()

        subject_id_str = f"sub-{int(subject_id[1:]):03d}"

        events_path = DATA_ROOT / "simulator" / subject_id_str / f"{subject_id_str}_events.csv"
        self._event_times = np.loadtxt(events_path, dtype=np.float64)
        if self._event_times.ndim == 0:
            self._event_times = np.array([float(self._event_times)])
        self._next_event_idx = 0

        self._pre_handled_for_upcoming_event = False
        self._pending_processed_pre: Optional[mne.EpochsArray] = None

        self.preprocessor = Preprocessor(str(FORWARD_PATH))
        self.dipole_fitter = DipoleFitter(str(FORWARD_PATH))
        self.normalizer = TEPNormalizer()

        global_backrotation = np.load(str(GLOBAL_BACKROTATION_PATH))
        self.predictor = OnlinePredictor(
            global_backrotation,
            model_path=str(PRETRAINED_MODEL_PATH),
            seed=SEED,
        )

        print(
            f"PRIME decider ready  subject={subject_id}  fs={sampling_frequency}  "
            f"eeg={num_eeg_channels}  emg={num_emg_channels}  "
            f"events={len(self._event_times)}"
        )

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_configuration(self) -> dict[str, Any]:
        # Periodic buffer ends at the current sample; with EVENT_LOOKAHEAD_SEC until the
        # pulse, that sample is raw_pre_tmax relative to the upcoming event.
        sample_window = [self._raw_pre_tmin + EVENT_LOOKAHEAD_SEC, 0.0]
        event_sample_window = [self._raw_post_tmin, self._raw_post_tmax]
        return {
            "periodic_processing_interval": 1.0 / self.sampling_frequency,
            "sample_window": sample_window,
            "event_sample_window": event_sample_window,
            "warm_up_rounds": 3,
        }

    # ==================================================================
    # Periodic processing (pre-stim, 5 ms before each event)
    # ==================================================================

    def _event_upcoming(self, reference_time: float) -> bool:
        if self._next_event_idx >= len(self._event_times):
            return False
        event_time = float(self._event_times[self._next_event_idx])
        dt = 1.0 / self.sampling_frequency
        return abs((event_time - reference_time) - EVENT_LOOKAHEAD_SEC) <= dt / 2

    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int,
            is_warm_up: bool) -> dict[str, Any] | None:
        if is_warm_up or not self._event_upcoming(reference_time):
            return None
        if self._pre_handled_for_upcoming_event:
            return None

        pre_checksum = hashlib.sha256(eeg_buffer.tobytes()).hexdigest()
        print(f"Pre-epoch sha256={pre_checksum}")

        self._pre_handled_for_upcoming_event = True

        if not self.is_calibrated:
            with _profile("add_raw_pre_epoch"):
                self.preprocessor.add_raw_pre_epoch(eeg_buffer)
            print(f"Calibration pre epoch queued for trial {self.trial_count + 1}/{N_CALIBRATION_TRIALS}")
            return None

        with _profile("preprocess_pre"):
            processed_pre = self.preprocessor.preprocess_pre(eeg_buffer)
        self._pending_processed_pre = processed_pre
        if processed_pre is None:
            print(f"Trial {self.trial_count + 1}: pre REJECTED by preprocessing")
            return None

        with _profile("predict"):
            probability = self.predictor.predict(processed_pre)
        print(f"Trial {self.trial_count + 1}: prediction={probability:.6f} (pre-stim)")
        return None

    # ==================================================================
    # Event processing (post-stim at TMS pulse)
    # ==================================================================

    def process_event(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:

        post_checksum = hashlib.sha256(eeg_buffer.tobytes()).hexdigest()
        print(f"Post-epoch sha256={post_checksum}")

        if not self.is_calibrated:
            with _profile("add_raw_post_epoch"):
                self.preprocessor.add_raw_post_epoch(eeg_buffer)
            self.trial_count += 1
            print(f"Calibration trial {self.trial_count}/{N_CALIBRATION_TRIALS}")

            if self.trial_count >= N_CALIBRATION_TRIALS:
                with _profile("run_calibration"):
                    self._run_calibration()
        else:
            self.trial_count += 1
            processed_pre = self._pending_processed_pre
            self._pending_processed_pre = None

            with _profile("preprocess_post"):
                processed_post = self.preprocessor.preprocess_post(eeg_buffer)
            if processed_pre is None or processed_post is None:
                print(f"Trial {self.trial_count}: REJECTED by preprocessing")
            else:
                with _profile("fit_trial"):
                    amplitude = self.dipole_fitter.fit_trial(processed_post)
                with _profile("normalize"):
                    label = self.normalizer.transform(amplitude)
                with _profile("finetune"):
                    self.predictor.finetune(processed_pre, label)
                print(f"Trial {self.trial_count}: label={label:.6f}")

        self._next_event_idx += 1
        self._pre_handled_for_upcoming_event = False
        return None

    # ==================================================================
    # Calibration
    # ==================================================================

    def _run_calibration(self) -> None:
        print("\n" + "=" * 60)
        print("RUNNING CALIBRATION")
        print("=" * 60)

        with _profile("preprocessor.calibrate"):
            cal_pre, cal_post = self.preprocessor.calibrate()
        print(f"  Preprocessor done: {len(cal_pre)} trials survived rejection")

        with _profile("dipole_fitter.calibrate"):
            amplitudes = self.dipole_fitter.calibrate(cal_post)
        print("  Dipole fitter calibrated")

        with _profile("normalizer.calibrate"):
            labels = self.normalizer.calibrate(amplitudes)
        print("  TEP normalizer calibrated")

        with _profile("predictor.calibrate"):
            self.predictor.calibrate(cal_pre, labels)
        print("  Predictor calibrated")

        self.is_calibrated = True
        print("=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60 + "\n")
