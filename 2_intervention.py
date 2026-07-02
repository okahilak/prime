"""PRIME intervention decider for NeuroSimo.

Implements protocols/prime_intervention.yaml (PRIME-TEP calibration + application):

  1. Calibration stage (100 predetermined high-ITI trials):
     Accumulate trials via process_pulse, then batch-calibrate in calibrate_prime task.
  2. Intervention blocks (4 x 200 trials): one factor with three conditions per trial
       - PERIODIC_TRIPLET       60%  (120/block)  PRIME-guided triplet (TBS)
       - PERIODIC_SINGLE        30%  ( 60/block)  PRIME-guided single pulse
       - PREDETERMINED_SINGLE   10%  ( 20/block)  predetermined single pulse (scheduled ITI)
     Conditions are balanced within each 20-trial mini-block, so the predetermined
     singles (used to track TEP amplitude trends over time) stay evenly spread.
     process_periodic: rolling-window QC + prediction; schedule pulse when excitable.
     process_pulse: post-stimulus preprocessing, dipole fit, tep_amplitude, finetune.

Calibration and intervention run in a single session so the calibrated (and then
finetuned) PRIME model persists in memory across both. Baseline and evaluation
are separate protocols/deciders.

Requires:
  - A pretrained PRIME model checkpoint (.pt)
  - A global back-rotation matrix (.npy)
  - An MNE forward solution (.fif) for dipole fitting
"""

import csv
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
from util.magventure_tms_mock import MockTMS

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

SEED = 201

# Wall-clock latencies (ms) collected per call and flushed to profile_*.csv once
# at teardown, so the hot path (process_periodic) never touches the filesystem.
PROFILE_METRICS = ("predict", "check_qc", "finetune", "process_pulse")


MINI_BLOCK_COMPOSITION = (
    ["prime_triplet"] * 12
    + ["prime_single_pulse"] * 6
    + ["predetermined_single"] * 2
)
assert len(MINI_BLOCK_COMPOSITION) == MINI_BLOCK_SIZE

# Open-loop session: 120 predetermined triplets + 80 predetermined singles per block
# (12 + 8 per mini-block), no PRIME-guided triggering.
OPEN_LOOP_MINI_BLOCK_COMPOSITION = (
    ["predetermined_triplet"] * 12
    + ["predetermined_single"] * 8
)
assert len(OPEN_LOOP_MINI_BLOCK_COMPOSITION) == MINI_BLOCK_SIZE


BUFFER_TOLERANCE = 0.005


def timed_ms(fn, /, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


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

        # Quality control window
        self.qc_tmin, self.qc_tmax = get_qc_time_range()
        self.qc_window_size = self.qc_tmax - self.qc_tmin

        self.post_initial_tmin, self.post_initial_tmax = get_post_initial_time_range()

        self.rng = np.random.default_rng(SEED + subject_id)

        self.tms = MagVentureTMS() if not self.mock_tms_device else MockTMS()

        self.is_calibrated = False
        self.current_pre: Optional[np.ndarray] = None
        self.current_is_forced = False
        self.prime_attempt_count = 0
        self.qc_fail_count = 0

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
        composition = OPEN_LOOP_MINI_BLOCK_COMPOSITION if self.is_open_loop_session else MINI_BLOCK_COMPOSITION
        n_mini_blocks = INTERVENTION_BLOCK_TRIALS // MINI_BLOCK_SIZE
        for block in range(1, 5):
            conditions: list[str] = []
            for _ in range(n_mini_blocks):
                mini_block = composition.copy()
                self.rng.shuffle(mini_block)
                conditions.extend(mini_block)
            self.intervention_conditions[f"intervention_block_{block}"] = conditions

        global_backrotation = np.load(GLOBAL_BACKROTATION_PATH)
        self.predictor = OnlinePredictor(global_backrotation, model_path=PRETRAINED_MODEL_PATH, seed=SEED)

        self.preprocessor = Preprocessor(FORWARD_PATH)
        self.dipole_fitter = DipoleFitter(FORWARD_PATH)
        self.normalizer = TEPNormalizer()

        # Create results directory and trials CSV file.
        self.results_dir = Path("results") / str(subject_id) / ("open_loop" if self.is_open_loop_session else "prime")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trials_csv = self.results_dir / "trials_intervention.csv"

        # Count of intervention trials completed since calibration; used to name
        # the per-trial classifier checkpoints (classifier_after_trial_<i>.pt).
        self.post_calibration_trial_count = 0
        if self.trials_csv.exists() and not self.overwrite_existing_results:
            raise FileExistsError(
                f"Intervention results already exist: {self.trials_csv} — "
                "the intervention may have already been run for this subject and session type. "
                "Delete the file or enable overwriting."
            )
        self.csv_fields = [
            "stage", "trial_in_stage", "condition", "iti",
            "trial_start_time", "target_time", "max_time",
            "trigger_time", "is_forced", "pulse_time",
            "preprocessing_failed", "postprocessing_failed",
            "prediction_probability", "prime_attempts", "qc_failures", "tep_amplitude",
        ]
        with open(self.trials_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writeheader()
        self.current_trial: dict = {}

        self.profile_samples: dict[str, list[float]] = {name: [] for name in PROFILE_METRICS}

        print(
            f"PRIME intervention decider ready  subject={subject_id}  fs={sampling_frequency}  "
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
    # Protocol helpers
    # ==================================================================
    @staticmethod
    def is_intervention_stage(stage_name: str) -> bool:
        return stage_name.startswith("intervention_block_")

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
        if stage_name != "calibration" and not self.is_intervention_stage(stage_name):
            raise ValueError("Incorrect protocol, must be 2_intervention.yaml for intervention")

        self.current_is_forced = False
        self.current_pre = None
        self.prime_attempt_count = 0
        self.qc_fail_count = 0
        self.trial_max_time = None

        iti = self.rng.uniform(ITI_MIN, ITI_MAX)

        self.current_trial = {
            "stage": stage_name,
            "trial_in_stage": trial_in_stage,
            "condition": None,
            "iti": None,
            "trial_start_time": start_time,
            "target_time": None,
            "max_time": None,
            "prediction_probability": None,
            "prime_attempts": None,
            "qc_failures": None,
            "trigger_time": None,
            "is_forced": None,
            "pulse_time": None,
            "tep_amplitude": None,
            "preprocessing_failed": False,
            "postprocessing_failed": False,
        }

        # Calibration: single pulses, predetermined.
        if stage_name == "calibration":
            self.tms.set_single_pulse(self.single_pulse_intensity)
            self.current_trial["iti"] = iti
            self.current_trial["target_time"] = start_time + iti
            return {"trigger_offset": iti}

        # Intervention: look up this trial's condition and set the matching pulse type.
        elif self.is_intervention_stage(stage_name):
            condition = self.condition_for_trial(stage_name, trial_in_stage)
            self.current_trial["condition"] = condition

            if condition == "prime_triplet":
                self.trial_max_time = start_time + iti
                self.current_trial["max_time"] = self.trial_max_time
                self.tms.set_tbs(self.tbs_intensity)
                return None

            elif condition == "prime_single_pulse":
                self.trial_max_time = start_time + iti
                self.current_trial["max_time"] = self.trial_max_time
                self.tms.set_single_pulse(self.single_pulse_intensity)
                return None

            elif condition == "predetermined_single":
                self.tms.set_single_pulse(self.single_pulse_intensity)
                self.current_trial["iti"] = iti
                self.current_trial["target_time"] = start_time + iti
                return {"trigger_offset": iti}

            elif condition == "predetermined_triplet":
                self.tms.set_tbs(self.tbs_intensity)
                self.current_trial["iti"] = iti
                self.current_trial["target_time"] = start_time + iti
                return {"trigger_offset": iti}

            else:
                raise ValueError(f"Unknown condition: {condition!r}")

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

        self.current_pre = self.preprocessor.preprocess_pre(eeg_buffer, time_offsets, from_pulse=False)

        self.qc_window_good.append(self.current_pre is not None)

        if reference_time > self.trial_max_time:
            print(f"Prime trial max time exceeded, triggering a pulse (prime_attempts={self.prime_attempt_count})")
            self.current_is_forced = True
            self.current_trial["prime_attempts"] = self.prime_attempt_count
            self.current_trial["qc_failures"] = self.qc_fail_count

            return {"trigger_offset": TRIGGER_OFFSET}

        qc_passes, qc_ms = timed_ms(self.check_qc)
        self.record_profile("check_qc", qc_ms)
        if not qc_passes:
            self.qc_fail_count += 1
            print(f"Quality control check rejected (qc_failures={self.qc_fail_count})")
            return None

        # QC passed, so the most recent window is good and pre is available.
        self.prime_attempt_count += 1
        probability, predict_ms = timed_ms(self.predictor.predict, self.current_pre)
        self.record_profile("predict", predict_ms)
        print(f"Prime prediction={probability:.3f}, prediction_time={predict_ms:.1f}ms, attempt={self.prime_attempt_count}")
        if probability < PREDICTION_THRESHOLD:
            return None

        print(f"Prime trigger scheduled after {self.prime_attempt_count} attempt(s)")

        self.current_trial["prediction_probability"] = probability
        self.current_trial["prime_attempts"] = self.prime_attempt_count
        self.current_trial["qc_failures"] = self.qc_fail_count
        self.current_trial["trigger_time"] = reference_time
        self.current_trial["target_time"] = reference_time + TRIGGER_OFFSET

        return {"trigger_offset": TRIGGER_OFFSET}

    # ==================================================================
    # Pulse processing
    # ==================================================================

    def process_pulse(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        t0 = time.perf_counter()

        tep_amplitude = None

        if stage_name == "calibration":
            self.preprocessor.add_trial(eeg_buffer, time_offsets)
            print(f"Calibration trial {trial_in_stage + 1} collected")

        elif self.is_intervention_stage(stage_name):
            tep_amplitude = self.process_intervention_pulse(time_offsets, eeg_buffer, stage_name, trial_in_stage)
            condition = self.current_trial.get("condition", "unknown")
            if tep_amplitude is not None:
                print(f"Intervention trial {trial_in_stage + 1} finished: condition={condition} TEP amplitude={tep_amplitude:.3f}")
            else:
                print(f"Intervention trial {trial_in_stage + 1} finished: condition={condition} TEP amplitude=failed")

        else:
            raise ValueError(f"Unknown stage: {stage_name!r}")

        self.current_trial.update({
            "pulse_time": reference_time,
            "is_forced": self.current_is_forced,
            "tep_amplitude": tep_amplitude,
        })
        self.write_trial_row()
        if self.is_intervention_stage(stage_name):
            self.post_calibration_trial_count += 1
            self.save_classifier_checkpoint(
                f"classifier_after_trial_{self.post_calibration_trial_count}.pt"
            )
        self.save_raw_buffers(
            stage_name, trial_in_stage,
            self.extract_raw_pre_from_pulse(time_offsets, eeg_buffer)[0],
            self.extract_raw_post_from_pulse(time_offsets, eeg_buffer),
        )

        self.record_profile("process_pulse", (time.perf_counter() - t0) * 1000)

    def process_intervention_pulse(
            self, time_offsets: np.ndarray, eeg_buffer: np.ndarray,
            stage_name: str, trial_in_stage: int) -> Optional[float]:
        condition = self.condition_for_trial(stage_name, trial_in_stage)

        success, tep_amplitude = self.analyze_tep(time_offsets, eeg_buffer)

        if not success:
            print("Trial failed: post-stimulus processing failed")
            self.current_trial["postprocessing_failed"] = True
            return None

        assert tep_amplitude is not None

        if condition == "prime_single_pulse":
            # For non-forced PRIME singles, pre is the prediction window from process_periodic.
            # For forced PRIME singles, pre is extracted from the pulse-aligned buffer.
            if self.current_is_forced:
                pre = self.preprocess_pre_from_pulse(time_offsets, eeg_buffer)
            else:
                pre = self.current_pre

            # If preprocessing fails (can only happen for forced trials), skip finetuning.
            if pre is not None:
                _, finetune_ms = timed_ms(self.predictor.finetune, pre, tep_amplitude)
                self.record_profile("finetune", finetune_ms)
            else:
                print("Single pulse PRIME trial pre-stimulus preprocessing failed, skipping finetuning")
                self.current_trial["preprocessing_failed"] = True

        elif condition == "prime_triplet":
            # Do not finetune on triplet trials.
            pass

        # Only train on predetermined trials on a prime session (not open loop session)
        elif condition == "predetermined_single":
            # PRIME session: open-loop singles are valid single-pulse trials → finetune.
            # Open-loop session: clean control → never finetune.
            if not self.is_open_loop_session:
                pre = self.preprocess_pre_from_pulse(time_offsets, eeg_buffer)
                if pre is not None:
                    _, finetune_ms = timed_ms(self.predictor.finetune, pre, tep_amplitude)
                    self.record_profile("finetune", finetune_ms)
                else:
                    print("Predetermined trial pre-stimulus preprocessing failed: skipping finetuning")
                    self.current_trial["preprocessing_failed"] = True

        elif condition == "predetermined_triplet":
            # Open-loop triplets: no finetuning.
            pass

        else:
            raise ValueError(f"Unknown condition: {condition!r}")

        return tep_amplitude

    # ==================================================================
    # Trial logging
    # ==================================================================

    def save_classifier_checkpoint(self, name: str) -> None:
        """Persist the (finetuned) classifier weights under ``name``, unless
        running open loop.

        In an open-loop session the model is never calibrated or finetuned, so
        there is nothing worth checkpointing.
        """
        if self.is_open_loop_session:
            return
        self.predictor.save_checkpoint(self.results_dir / name)

    def write_trial_row(self) -> None:
        row = {field: self.current_trial.get(field) for field in self.csv_fields}
        with open(self.trials_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writerow(row)

    # ==================================================================
    # Profiling
    # ==================================================================

    def record_profile(self, name: str, elapsed_ms: float) -> None:
        """Buffer a single latency sample (ms) in memory. Cheap: no I/O, safe on
        the hot path. Flushed to disk once by write_profile at teardown."""
        self.profile_samples[name].append(elapsed_ms)

    def write_profile(self) -> None:
        """Write each metric's collected latencies to results/.../profile_<name>.csv,
        one sample per line under a `duration_ms` header."""
        for name, samples in self.profile_samples.items():
            path = self.results_dir / f"profile_{name}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["duration_ms"])
                for elapsed_ms in samples:
                    writer.writerow([f"{elapsed_ms:.3f}"])

    def __del__(self) -> None:
        self.write_profile()

    # ==================================================================
    # TEP analysis
    # ==================================================================

    def preprocess_pre_from_pulse(
            self, time_offsets: np.ndarray, eeg_buffer: np.ndarray
    ) -> Optional[np.ndarray]:
        pre_buffer, pre_time_offsets = self.extract_raw_pre_from_pulse(time_offsets, eeg_buffer)
        return self.preprocessor.preprocess_pre(pre_buffer, pre_time_offsets, from_pulse=True)

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
        tep_amplitude = self.normalizer.transform(amplitude)

        return True, tep_amplitude

    # ==================================================================
    # Calibration
    # ==================================================================

    def run_calibration(self) -> int:
        print("Running calibration...")

        t0 = time.perf_counter()

        model_buffers, dipole_buffers = self.preprocessor.calibrate()
        amplitudes = self.dipole_fitter.calibrate(dipole_buffers)
        tep_amplitudes = self.normalizer.calibrate(amplitudes)
        self.predictor.calibrate(model_buffers, tep_amplitudes)
        self.predictor.warm_up()

        # Number of calibration trials that survived preprocessing and were
        # actually used to fit the PRIME model.
        num_calibration_trials = int(model_buffers.shape[0])

        print(f"Calibration took {time.perf_counter() - t0:.2f} seconds")
        print(f"Calibration used {num_calibration_trials} trials")
        self.write_calibration_summary(num_calibration_trials)

        self.is_calibrated = True
        self.save_classifier_checkpoint("classifier_after_calibration.pt")
        return num_calibration_trials

    def write_calibration_summary(self, num_calibration_trials: int) -> None:
        """Record the number of trials used for calibration, separate from the trials CSV."""
        summary_path = self.results_dir / "calibration_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["num_calibration_trials"])
            writer.writeheader()
            writer.writerow({"num_calibration_trials": num_calibration_trials})
