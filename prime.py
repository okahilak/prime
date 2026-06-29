"""PRIME decider module for NeuroSimo.

Implements protocols/prime.yaml (PRIME-TEP validation):

  1. Calibration stage (100 predetermined high-ITI trials):
     Accumulate trials via process_pulse, then batch-calibrate in calibrate_prime task.
  2. Intervention blocks (4 x 200 trials): one factor with three conditions per trial
       - PERIODIC_TRIPLET       60%  (120/block)  PRIME-guided triplet (TBS)
       - PERIODIC_SINGLE        30%  ( 60/block)  PRIME-guided single pulse
       - PREDETERMINED_SINGLE   10%  ( 20/block)  predetermined single pulse (scheduled ITI)
     Conditions are balanced within each 20-trial mini-block, so the predetermined
     singles (used to track TEP amplitude trends over time) stay evenly spread.
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
from util.magventure_tms import MagVentureTMS

# ---------------------------------------------------------------------------
# Paths — adjust per setup
# ---------------------------------------------------------------------------

FORWARD_PATH = Path("offline_data") / "fsaverage" / "fsaverage-fwd.fif"

# Use a model trained on all subjects
PRETRAINED_MODEL_PATH = Path("offline_results") / "train" / "pretrained_all.pt"
GLOBAL_BACKROTATION_PATH = Path("offline_results") / "train" / "global_backrotation_all.npy"

# ---------------------------------------------------------------------------
# Protocol parameters
# ---------------------------------------------------------------------------

PREDICTION_THRESHOLD = 0.5
TRIGGER_OFFSET = 0.01

# TODO: Change for each subject to match the protocol specifications.
AMPLITUDE_SINGLE_PULSE = 50   # % MSI, used for baseline / calibration / evaluation / single-pulse intervention
AMPLITUDE_TBS = 60            # % MSI, used for intervention triplets

ITI_MIN = 2.5
ITI_MAX = 5.5

# Intervention block structure (PRIME application session).
INTERVENTION_BLOCK_TRIALS = 200
MINI_BLOCK_SIZE = 20

# Pre-stimulus QC. process_periodic runs every 10 ms, so each call contributes
# one 200 ms window's good/bad result. A candidate stimulation is allowed only if
#   1. the most recent 5 windows are all good, and
#   2. at least 80% of the windows in the last 500 ms (50 windows) are good.
QC_RECENT_WINDOWS = 5
QC_HISTORY_WINDOWS = 50        # 500 ms / 10 ms step
QC_MIN_GOOD_FRACTION = 0.80

SEED = 42


MINI_BLOCK_COMPOSITION = (
    ["prime_triplet"] * 12
    + ["prime_single_pulse"] * 6
    + ["predetermined"] * 2
)
assert len(MINI_BLOCK_COMPOSITION) == MINI_BLOCK_SIZE


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

        self.tms = MagVentureTMS()

        self.is_calibrated = False
        self.current_pre: Optional[np.ndarray] = None
        self.current_is_forced = False

        self.trial_max_time = None

        # Rolling good/bad result of the most recent pre-stimulus QC windows.
        # Holds the last 500 ms (one entry per 10 ms periodic call); older
        # entries — including any post-pulse windows from the previous trial —
        # fall out on their own.
        self.qc_window_good: deque[bool] = deque(maxlen=QC_HISTORY_WINDOWS)

        # Pre-compute the per-trial condition schedule for each intervention
        # block. Done once here (deterministic, seeded) so that querying a
        # trial's condition later — including arming the next trial early — is a
        # pure lookup with no effect on the RNG stream.
        self.intervention_conditions: dict[str, list[str]] = {}
        n_mini_blocks = INTERVENTION_BLOCK_TRIALS // MINI_BLOCK_SIZE
        for block in range(1, 5):
            conditions: list[str] = []
            for _ in range(n_mini_blocks):
                mini_block = MINI_BLOCK_COMPOSITION.copy()
                self.rng.shuffle(mini_block)
                conditions.extend(mini_block)
            self.intervention_conditions[f"intervention_block_{block}"] = conditions

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
    @staticmethod
    def is_intervention_stage(stage_name: str) -> bool:
        return stage_name.startswith("intervention_block_")

    @staticmethod
    def is_evaluation_stage(stage_name: str) -> bool:
        return stage_name.startswith("evaluation_")

    def condition_for_trial(self, stage_name: str, trial_in_stage: int) -> str:
        """The intervention condition for a given trial. Pure lookup."""
        return self.intervention_conditions[stage_name][trial_in_stage]

    def check_qc(self) -> bool:
        """Whether the recent pre-stimulus windows allow stimulation."""
        history = self.qc_window_good
        if len(history) < QC_RECENT_WINDOWS:
            return False

        recent = list(history)

        # 1. The most recent QC_RECENT_WINDOWS windows must all be good.
        if not all(recent[-QC_RECENT_WINDOWS:]):
            return False

        # 2. At least QC_MIN_GOOD_FRACTION of the last 500 ms must be good.
        return sum(recent) / len(recent) >= QC_MIN_GOOD_FRACTION

    # ==================================================================
    # Trial preparation (timing + stimulator arming)
    # ==================================================================

    def prepare_trial(self, start_time: float, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        """Arm the stimulator for the upcoming trial and, for predetermined
        trials, schedule the trigger upfront by returning its ITI.

        Returns None for PRIME-guided (periodic) trials, whose trigger is
        scheduled later by process_periodic.
        """
        self.current_is_forced = False
        self.current_pre = None
        self.trial_max_time = None

        iti = self.rng.uniform(ITI_MIN, ITI_MAX)

        # Baseline: single pulses, predetermined.
        if stage_name == "baseline":
            self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
            return {"trigger_offset": iti}

        # Calibration: single pulses, predetermined.
        elif stage_name == "calibration":
            self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
            return {"trigger_offset": iti}

        # Intervention: look up this trial's condition and set the matching pulse type.
        elif self.is_intervention_stage(stage_name):
            condition = self.condition_for_trial(stage_name, trial_in_stage)
            if condition == "prime_triplet":
                self.trial_max_time = start_time + iti
                self.tms.set_tbs(AMPLITUDE_TBS)
                return None

            elif condition == "prime_single_pulse":
                self.trial_max_time = start_time + iti
                self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
                return None

            elif condition == "predetermined":
                self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
                return {"trigger_offset": iti}

            else:
                raise ValueError(f"Unknown condition: {condition!r}")

        # Evaluation: single pulses, predetermined.
        elif self.is_evaluation_stage(stage_name):
            self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
            return {"trigger_offset": iti}

        else:
            raise ValueError(f"Unknown stage: {stage_name!r}")

    # ==================================================================
    # Calibration task
    # ==================================================================

    def process_task(self, task_name: str) -> None:
        if task_name == "calibrate_prime":
            self.run_calibration()
            return

        raise ValueError(f"Unknown task: {task_name!r}")

    # ==================================================================
    # Periodic processing (PRIME-guided intervention)
    # ==================================================================

    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int,
            is_warm_up: bool) -> dict[str, Any] | None:

        self.current_pre = self.preprocessor.preprocess_pre(eeg_buffer, time_offsets, online=True)

        self.qc_window_good.append(self.current_pre is not None)

        if reference_time > self.trial_max_time:
            print(f"Prime trial max time exceeded, triggering a pulse")
            self.current_is_forced = True

            return {"trigger_offset": TRIGGER_OFFSET}

        qc_passes = self.check_qc()
        if not qc_passes:
            print(f"Quality control check rejected")
            return None

        # QC passed, so the most recent window is good and pre is available.
        probability, predict_ms = timed_ms(self.predictor.predict, self.current_pre)
        print(f"Prime prediction={probability:.3f}, prediction_time={predict_ms:.1f}ms")
        if probability < PREDICTION_THRESHOLD:
            return None

        print("Prime trigger scheduled")

        return {"trigger_offset": TRIGGER_OFFSET}

    # ==================================================================
    # Pulse processing
    # ==================================================================

    def process_pulse(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:

        if stage_name == "baseline":
            return None

        elif stage_name == "calibration":
            self.preprocessor.add_trial(eeg_buffer, time_offsets)
            print(f"Calibration trial {trial_in_stage + 1} collected")
            return None

        elif self.is_intervention_stage(stage_name):
            return self.process_intervention_pulse(time_offsets, eeg_buffer, stage_name, trial_in_stage)

        elif self.is_evaluation_stage(stage_name):
            return None

        else:
            raise ValueError(f"Unknown stage: {stage_name!r}")

    def process_intervention_pulse(
            self, time_offsets: np.ndarray, eeg_buffer: np.ndarray,
            stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        condition = self.condition_for_trial(stage_name, trial_in_stage)

        success, label = self.analyze_tep(time_offsets, eeg_buffer)

        if not success:
            print("Trial failed: post-stimulus processing failed")
            return None

        assert label is not None

        if condition == "predetermined":
            # Predetermined trials skip process_periodic, so their pre-stimulus
            # window is extracted here from the pulse-aligned buffer.
            pre = self.preprocess_pre_from_pulse(time_offsets, eeg_buffer)

            if pre is None:
                print("Predetermined trial pre-stimulus preprocessing failed: skipping finetuning")
                return None

            self.predictor.finetune(pre, label)

        elif condition == "prime_single_pulse":
            # For non-forced PRIME singles, pre is the prediction window from process_periodic.
            # For forced PRIME singles, pre is extracted from the pulse-aligned buffer.
            if self.current_is_forced:
                pre = self.preprocess_pre_from_pulse(time_offsets, eeg_buffer)
            else:
                pre = self.current_pre

            # If preprocessing fails (can only happen for forced trials), skip finetuning.
            if pre is None:
                print("Single pulse PRIME trial pre-stimulus preprocessing failed, skipping finetuning")
                return None

            self.predictor.finetune(pre, label)

        elif condition == "prime_triplet":
            # Do not finetune on triplet trials.
            pass

        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        print(f"Trial {trial_in_stage + 1} ({stage_name}) finished: label={label:.3f}")

        return None

    # ==================================================================
    # TEP analysis
    # ==================================================================

    def preprocess_pre_from_pulse(
            self, time_offsets: np.ndarray, eeg_buffer: np.ndarray
    ) -> Optional[np.ndarray]:
        """Extract and preprocess the pre-stimulus QC window from a pulse-aligned buffer."""
        pre_buffer, pre_time_offsets = crop_eeg_buffer(
            eeg_buffer,
            time_offsets,
            self.qc_tmin,
            self.qc_tmax,
        )
        return self.preprocessor.preprocess_pre(pre_buffer, pre_time_offsets, online=True)

    def analyze_tep(
            self, time_offsets: np.ndarray, eeg_buffer: np.ndarray
    ) -> tuple[bool, Optional[float]]:
        post_buffer, post_time_offsets = crop_eeg_buffer(
            eeg_buffer,
            time_offsets,
            self.post_initial_tmin,
            self.post_initial_tmax,
        )
        post = self.preprocessor.preprocess_post(post_buffer, post_time_offsets)

        if post is None:
            return False, None

        amplitude = self.dipole_fitter.fit_trial(post)
        label = self.normalizer.transform(amplitude)

        return True, label

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
