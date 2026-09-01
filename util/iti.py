"""Inter-trial interval sampling for predetermined (open-loop) trials.

Imported by 1_baseline.py, 2_intervention.py and 3_evaluation.py so that every
predetermined trial in every stage draws from this distribution.

To change the distribution, change sample_iti() and the two bounds. Nothing else
in the deciders needs to move.
"""
import numpy as np

# Support of the distribution. ITI_MIN must stay above the protocol yaml's
# execution.minimum_trial_interval, which NeuroSimo enforces independently.
ITI_MIN = 2.5
ITI_MAX = 5.5


def sample_iti(rng: np.random.Generator) -> float:
    """Draw one ITI in seconds. rng is the decider's seeded Generator, so the
    draw stays inside the session's reproducible stream."""
    return float(rng.uniform(ITI_MIN, ITI_MAX))


def _self_test(n: int = 100_000, seed: int = 0) -> None:
    """Fail at import, not mid-session, if sample_iti can leave its bounds.

    A too-short ITI is a stimulator-safety matter, so this is deliberately a
    hard failure rather than a clamp.
    """
    rng = np.random.default_rng(seed)
    s = np.array([sample_iti(rng) for _ in range(n)])
    if s.min() < ITI_MIN or s.max() > ITI_MAX:
        raise ValueError(
            f"sample_iti() produced values outside [{ITI_MIN}, {ITI_MAX}]: "
            f"observed [{s.min():.3f}, {s.max():.3f}] over {n} draws."
        )


_self_test()