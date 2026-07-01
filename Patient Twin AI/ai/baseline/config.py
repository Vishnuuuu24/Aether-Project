"""Baseline-engine configuration (docs/05 §4-5).

All numeric defaults here are **statistical parameters given in docs/05** (28-day
window, 7-day EWMA half-life, min_n=50, min_days=7, 1.4826 MAD scale, z-buckets
2/3/4.5) — not clinical thresholds. Clinical values (SQI thresholds, population
ranges) live in `config/clinical/` and are injected, never defaulted here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class BaselineConfig:
    window_days: int = 28
    ewma_half_life_days: float = 7.0
    min_n: int = 50
    min_days: int = 7
    mad_scale: float = 1.4826
    # |z_robust| magnitude bucket edges (docs/05 §5).
    mild_z: float = 2.0
    moderate_z: float = 3.0
    marked_z: float = 4.5
    # Strongly-circadian metrics stratified by time-of-day when the bucket has its
    # own sufficiency (docs/05 §4: "Sleep, resting HR, and temperature").
    circadian_metrics: frozenset[str] = field(
        default_factory=lambda: frozenset({"heart_rate", "skin_temp", "sleep"})
    )


def circadian_bucket(moment: datetime) -> str:
    """Coarse time-of-day bucket for circadian stratification (docs/05 §4).

    v1 uses the timestamp's own tz hour; patient-local bucketing is a later
    refinement. Structural binning only — no clinical content.
    """
    hour = moment.hour
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"
