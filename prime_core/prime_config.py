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
    """Full raw trial window covering all online preprocessing steps."""
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


def get_post_initial_time_range() -> tuple[float, float]:
    """Raw post-stimulus crop window [post_initial_tmin, post_initial_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.post_initial_tmin), float(cfg.post_initial_tmax)


def get_post_time_range() -> tuple[float, float]:
    """Post-stimulus processing window [post_tmin, post_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.post_tmin), float(cfg.post_tmax)


def get_raw_sfreq() -> float:
    """Raw EEG sampling rate (Hz) from the acquisition pipeline."""
    cfg = load_prime_config()
    return float(cfg.raw_sfreq)


def get_processed_sfreq() -> float:
    """Sampling rate (Hz) of the QC / model (pre-stimulus) window -- the PRIME input."""
    cfg = load_prime_config()
    return float(cfg.processed_sfreq)


def get_ica_sfreq() -> float:
    """Sampling rate (Hz) of the calibration ICA window (falls back to processed_sfreq)."""
    cfg = load_prime_config()
    return float(cfg.get("ica_sfreq", cfg.processed_sfreq))


def get_post_sfreq() -> float:
    """Sampling rate (Hz) of the post-stimulus window (falls back to processed_sfreq)."""
    cfg = load_prime_config()
    return float(cfg.get("post_sfreq", cfg.processed_sfreq))


def get_dipole_window_limits() -> tuple[float, float]:
    """Min / max dipole sub-window length in seconds (defaults = 3 / 6 samples at 1 kHz)."""
    cfg = load_prime_config()
    return (float(cfg.get("dipole_min_window", 0.003)),
            float(cfg.get("dipole_max_window", 0.006)))


def time_to_sample(time: float, first_time: float, sfreq: float) -> int:
    """Sample index for ``time`` on a uniform grid starting at ``first_time``."""
    return int(round((time - first_time) * sfreq))


def epoch_n_times(tmin: float, tmax: float, sfreq: float) -> int:
    """Number of samples in an inclusive epoch window at ``sfreq``."""
    return int(round((tmax - tmin) * sfreq)) + 1
