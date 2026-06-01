"""Load shared settings from configs/prime.yaml."""

from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

PRIME_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "prime.yaml"


@lru_cache(maxsize=1)
def load_prime_config():
    return OmegaConf.load(PRIME_CONFIG_PATH)


def get_pre_epoch_time_range() -> tuple[float, float]:
    """Pre-stimulus epoch window [pre_epoch_tmin, pre_epoch_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.pre_epoch_tmin), float(cfg.pre_epoch_tmax)


def get_post_epoch_time_range() -> tuple[float, float]:
    """Post-stimulus epoch window [post_epoch_tmin, post_epoch_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.post_epoch_tmin), float(cfg.post_epoch_tmax)


def get_raw_pre_epoch_time_range() -> tuple[float, float]:
    """Pre-stimulus crop window covering all online preprocessing steps (ICA, filtering, etc.)."""
    cfg = load_prime_config()
    return float(cfg.raw_pre_epoch_tmin), float(cfg.raw_pre_epoch_tmax)


def get_raw_post_epoch_time_range() -> tuple[float, float]:
    """Raw post-stimulus crop window [raw_post_epoch_tmin, raw_post_epoch_tmax] in seconds."""
    cfg = load_prime_config()
    return float(cfg.raw_post_epoch_tmin), float(cfg.raw_post_epoch_tmax)


def epoch_n_times(tmin: float, tmax: float, sfreq: float) -> int:
    """Number of samples in an inclusive epoch window at ``sfreq``."""
    return int(round((tmax - tmin) * sfreq)) + 1
