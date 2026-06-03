"""PRIME decider module for NeuroSimo.

Implements the PRIME online pipeline equivalent to simulate_online.py:
  1. Calibration phase (first N_CALIBRATION_TRIALS events):
     Accumulate full calibration-window trials via process_event, then batch-calibrate.
  2. Intervention phase (remaining events):
     process_periodic: preprocess pre + predict at end of QC pre-stim window.
     process_event: crop post_initial window, preprocess post, fit dipole, label, finetune.

Requires:
  - A pretrained PRIME model checkpoint (.pt)
  - A global back-rotation matrix (.npy)
  - An MNE forward solution (.fif) for dipole fitting

See simulate_online.py for the offline simulation equivalent.
"""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from prime.online_predictor import OnlinePredictor
from prime.online_preprocessing.preprocessor import Preprocessor, crop_eeg_buffer
from prime.online_preprocessing.dipole_fitter import DipoleFitter
from prime.prime_config import (
    get_calibration_time_range,
    get_qc_time_range,
    get_post_initial_time_range,
)
from prime.tep_normalizer import TEPNormalizer

# ---------------------------------------------------------------------------
# Paths — adjust per setup
# ---------------------------------------------------------------------------

FORWARD_PATH = Path("data") / "fsaverage" / "fsaverage-fwd.fif"
PRETRAINED_MODEL_PATH = Path("results") / "train" / "pretrained.pt"
GLOBAL_BACKROTATION_PATH = Path("results") / "train" / "global_backrotation.npy"

# ---------------------------------------------------------------------------
# Protocol parameters
# ---------------------------------------------------------------------------

N_CALIBRATION_TRIALS = 125
SEED = 42


@contextmanager
def profile(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[profile] {label}: {elapsed * 1000:.1f}ms")


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

        self.trial_count = 0
        self.is_calibrated = False

        self.calibration_tmin, self.calibration_tmax = get_calibration_time_range()
        self.qc_tmin, self.qc_tmax = get_qc_time_range()
        self.event_lookahead = -self.qc_tmax
        self.post_initial_tmin, self.post_initial_tmax = get_post_initial_time_range()

        subject_id_str = f"sub-{subject_id:03d}"

        events_path = Path("data") / "simulator" / subject_id_str / f"{subject_id_str}_events.csv"
        all_event_times = np.loadtxt(events_path, dtype=np.float64)
        if all_event_times.ndim == 0:
            all_event_times = np.array([float(all_event_times)])
        self.event_times = all_event_times[N_CALIBRATION_TRIALS:]
        self.next_event_idx = 0

        self.pending_pre: Optional[np.ndarray] = None

        self.preprocessor = Preprocessor(FORWARD_PATH)
        self.dipole_fitter = DipoleFitter(FORWARD_PATH)
        self.normalizer = TEPNormalizer()

        global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)
        self.predictor = OnlinePredictor(
            global_backrotation,
            model_path=PRETRAINED_MODEL_PATH,
            seed=SEED,
        )

        print(
            f"PRIME decider ready  subject={subject_id}  fs={sampling_frequency}  "
            f"eeg={num_eeg_channels}  emg={num_emg_channels}  "
            f"events={len(all_event_times)}  intervention_events={len(self.event_times)}"
        )

    # ==================================================================
    # Configuration
    # ==================================================================

    def get_configuration(self) -> dict[str, Any]:
        sample_window = [self.qc_tmin - self.qc_tmax, 0.0]
        event_sample_window = [self.calibration_tmin, self.calibration_tmax]
        return {
            "periodic_processing_interval": 1.0 / self.sampling_frequency,
            "sample_window": sample_window,
            "event_sample_window": event_sample_window,
            "warm_up_rounds": 0,
        }

    # ==================================================================
    # Periodic processing (pre-stim, at QC window end before each event)
    # ==================================================================

    def event_upcoming(self, reference_time: float) -> bool:
        if self.next_event_idx >= len(self.event_times):
            return False

        event_time = float(self.event_times[self.next_event_idx])
        dt = 1.0 / self.sampling_frequency

        is_upcoming = abs((event_time - reference_time) - self.event_lookahead) <= dt / 2
        if is_upcoming:
            self.next_event_idx += 1

        return is_upcoming

    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int,
            is_warm_up: bool) -> dict[str, Any] | None:

        if not self.is_calibrated:
            return None

        if not self.event_upcoming(reference_time):
            return None

        with profile("preprocess_pre"):
            pre = self.preprocessor.preprocess_pre(eeg_buffer, time_offsets)

        self.pending_pre = pre
        if pre is None:
            print(f"Trial {self.trial_count + 1}: pre REJECTED by preprocessing")
            return None

        with profile("predict"):
            probability = self.predictor.predict(pre)
        print(f"Trial {self.trial_count + 1}: prediction={probability:.6f} (pre-stim)")
        return None

    # ==================================================================
    # Event processing (post-stim at TMS pulse)
    # ==================================================================

    def process_event(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:

        if not self.is_calibrated:
            self.preprocessor.add_trial(eeg_buffer, time_offsets)

            self.trial_count += 1
            print(f"Calibration trial {self.trial_count}/{N_CALIBRATION_TRIALS}")

            if self.trial_count == N_CALIBRATION_TRIALS:
                self.run_calibration()

            return None

        self.trial_count += 1
        pre = self.pending_pre
        self.pending_pre = None

        post_buffer, post_time_offsets = crop_eeg_buffer(
            eeg_buffer,
            time_offsets,
            self.post_initial_tmin,
            self.post_initial_tmax,
        )
        post = self.preprocessor.preprocess_post(post_buffer, post_time_offsets)

        if pre is None or post is None:
            print(f"Trial {self.trial_count}: REJECTED by preprocessing")
            return None

        amplitude = self.dipole_fitter.fit_trial(post)
        label = self.normalizer.transform(amplitude)
        self.predictor.finetune(pre, label)

        print(f"Trial {self.trial_count}: label={label:.6f}")

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

        print(f"Calibration took {time.perf_counter() - t0:.2f} seconds")
        self.is_calibrated = True
