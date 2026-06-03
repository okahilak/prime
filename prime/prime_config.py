"""Load shared settings from configs/prime.yaml."""

from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

PRIME_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prime.yaml"


@lru_cache(maxsize=1)
def load_prime_config():
    return OmegaConf.load(PRIME_CONFIG_PATH)


def get_model_time_range() -> tuple[float, float]:
    """Pre-stimulus epoch window [model_tmin, model_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.model_tmin), float(cfg.model_tmax)


def get_dipole_time_range() -> tuple[float, float]:
    """Post-stimulus epoch window [dipole_tmin, dipole_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.dipole_tmin), float(cfg.dipole_tmax)


def get_calibration_time_range() -> tuple[float, float]:
    """Pre-stimulus crop window covering all online preprocessing steps (ICA, filtering, etc.)."""
    cfg = load_prime_config()
    return float(cfg.calibration_tmin), float(cfg.calibration_tmax)


def get_ica_time_range() -> tuple[float, float]:
    """Pre-stimulus epoch window [ica_tmin, ica_tmax] used for ICA calibration."""
    cfg = load_prime_config()
    return float(cfg.ica_tmin), float(cfg.ica_tmax)


def get_qc_time_range() -> tuple[float, float]:
    """Pre-stimulus quality control window [qc_tmin, qc_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.qc_tmin), float(cfg.qc_tmax)


def get_post_time_range() -> tuple[float, float]:
    """Raw post-stimulus crop window [post_tmin, post_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.post_tmin), float(cfg.post_tmax)


def get_trial_time_range() -> tuple[float, float]:
    """Full raw trial window covering both pre- and post-stimulus crops."""
    calibration_tmin, calibration_tmax = get_calibration_time_range()
    post_tmin, post_tmax = get_post_time_range()
    return min(calibration_tmin, post_tmin), max(calibration_tmax, post_tmax)


def get_raw_sfreq() -> float:
    """Raw EEG sampling rate (Hz) from the acquisition pipeline."""
    cfg = load_prime_config()
    return float(cfg.raw_sfreq)


def get_processed_sfreq() -> float:
    """Sampling rate (Hz) after online preprocessing resampling."""
    cfg = load_prime_config()
    return float(cfg.processed_sfreq)


def time_to_sample(time: float, first_time: float, sfreq: float) -> int:
    """Sample index for ``time`` on a uniform grid starting at ``first_time``."""
    return int(round((time - first_time) * sfreq))


def epoch_n_times(tmin: float, tmax: float, sfreq: float) -> int:
    """Number of samples in an inclusive epoch window at ``sfreq``."""
    return int(round((tmax - tmin) * sfreq)) + 1
