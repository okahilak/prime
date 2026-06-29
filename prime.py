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

HIGH_ITI_MIN = 4.0
HIGH_ITI_MAX = 9.0
LOW_ITI_MIN = 2.5
LOW_ITI_MAX = 3.5

# Intervention block structure (PRIME application session).
INTERVENTION_BLOCK_TRIALS = 200
MINI_BLOCK_SIZE = 20

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
        self.pending_pre: Optional[np.ndarray] = None
        self.last_qc_failure_time: float = -np.inf

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

    # ==================================================================
    # Trial preparation (timing + stimulator arming)
    # ==================================================================

    def prepare_trial(self, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        """Arm the stimulator for the upcoming trial and, for predetermined
        trials, schedule the trigger upfront by returning its ITI.

        Returns None for PRIME-guided (periodic) trials, whose trigger is
        scheduled later by process_periodic.
        """
        # Baseline / evaluation: single pulses, low ITI, brain-state-independent.
        if stage_name == "baseline" or self.is_evaluation_stage(stage_name):
            self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
            return {"trigger_offset": self.rng.uniform(LOW_ITI_MIN, LOW_ITI_MAX)}

        # Calibration: single pulses, high ITI, brain-state-independent.
        if stage_name == "calibration":
            self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
            return {"trigger_offset": self.rng.uniform(HIGH_ITI_MIN, HIGH_ITI_MAX)}

        # Intervention: look up this trial's condition and set the matching pulse type.
        if self.is_intervention_stage(stage_name):
            condition = self.condition_for_trial(stage_name, trial_in_stage)
            if condition == "prime_triplet":
                self.tms.set_tbs(AMPLITUDE_TBS)
                return None

            elif condition == "prime_single_pulse":
                self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
                return None

            elif condition == "predetermined":
                self.tms.set_single_pulse(AMPLITUDE_SINGLE_PULSE)
                return {"trigger_offset": self.rng.uniform(HIGH_ITI_MIN, HIGH_ITI_MAX)}

            else:
                raise ValueError(f"Unknown condition: {condition!r}")

        return None

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

        if not self.is_calibrated:
            return None

        if not self.is_intervention_stage(stage_name):
            return None

        if self.condition_for_trial(stage_name, trial_in_stage) == "predetermined":
            return None

        pre = self.preprocessor.preprocess_pre(eeg_buffer, time_offsets, online=True)
        if pre is None:
            print(f"Periodic check t={reference_time:.3f}s: REJECTED (pre-stimulus)")
            self.last_qc_failure_time = reference_time
            return None

        if reference_time - self.last_qc_failure_time < 1.0:
            print(f"Periodic check t={reference_time:.3f}s: SUPPRESSED (QC failure within past 1 s)")
            return None

        probability, predict_ms = timed_ms(self.predictor.predict, pre)
        print(
            f"Periodic check t={reference_time:.3f}s: prediction={probability:.3f}  "
            f"prediction_time={predict_ms:.1f}ms"
        )
        if probability < PREDICTION_THRESHOLD:
            return None

        self.pending_pre = pre
        print(f"Trigger scheduled at t={reference_time + TRIGGER_OFFSET:.3f}s")

        return {"trigger_offset": TRIGGER_OFFSET}

    # ==================================================================
    # Pulse processing
    # ==================================================================

    def process_pulse(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:

        if stage_name == "baseline" or self.is_evaluation_stage(stage_name):
            return None

        if stage_name == "calibration":
            self.preprocessor.add_trial(eeg_buffer, time_offsets)
            print(f"Calibration trial {trial_in_stage + 1} collected")
            return None

        if self.is_intervention_stage(stage_name):
            # pending_pre is set only by process_periodic, so it is present for
            # PRIME-guided (periodic) trials and None for predetermined trials.
            pre = self.pending_pre
            self.pending_pre = None

            success, label = self.analyze_tep(time_offsets, eeg_buffer)

            # TODO (next step): implement the finetuning rule from the protocol.
            # Finetune PRIME only on *valid single-pulse* trials — PRIME-triggered,
            # predetermined, and forced singles — and never on triplets. One thing
            # this needs that isn't here yet:
            #   (1) Predetermined singles must also finetune, but they
            #       have no `pre` because they skip process_periodic. We need to
            #       compute/store a prediction for them at fire time.
            # For now we keep the prior behaviour: finetune whenever a PRIME
            # prediction is available (i.e. periodic trials only). This still
            # (incorrectly) finetunes periodic triplets — fixed in the next step.
            if success and pre is not None:
                self.predictor.finetune(pre, label)

            if not success:
                print(f"Trial {trial_in_stage + 1} ({stage_name}) failed: post-stimulus processing failed")
            else:
                print(f"Trial {trial_in_stage + 1} ({stage_name}) finished: label={label:.3f}")

            return {
                "trial_invalid": not success,
            }

        return None

    # ==================================================================
    # TEP analysis
    # ==================================================================

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