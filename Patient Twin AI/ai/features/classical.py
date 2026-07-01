"""ClassicalFeatureExtractor — v1 statistical-first `FeatureExtractor` (docs/05 §3-4).

Computes descriptive statistics over the quality-passing (SQI) readings in a
window. No clinical interpretation happens here — deviation scoring is the
BaselineEngine's job (docs/05 §5, §8). Non-scalar metrics (e.g. `bp`, `sleep`,
whose values are dicts) are counted but produce no numeric features in v1.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable

from schemas.features import FeatureSet, SignalWindow
from schemas.reading import Reading

from .sqi import SqiGate

FEATURE_EXTRACTOR_VERSION = "classical-v1"


class ClassicalFeatureExtractor:
    """Implements the `FeatureExtractor` protocol (docs/02 §6)."""

    def __init__(self, gate: SqiGate, *, version: str = FEATURE_EXTRACTOR_VERSION) -> None:
        self._gate = gate
        self._version = version

    def extract(self, window: SignalWindow) -> FeatureSet:
        passing = [reading for reading in window.readings if self._gate.passes(reading)]
        values = _numeric_values(passing)
        return FeatureSet(
            patient_id=window.patient_id,
            metric_code=window.metric_code,
            context=window.context,
            window_start=window.window_start,
            window_end=window.window_end,
            n_total=len(window.readings),
            n_quality_passing=len(passing),
            sqi_threshold_applied=self._gate.threshold_for(window.metric_code.value),
            features=_descriptive_stats(values) if values else {},
            feature_extractor_version=self._version,
        )


def _numeric_values(readings: Iterable[Reading]) -> list[float]:
    out: list[float] = []
    for reading in readings:
        value = reading.value
        if isinstance(value, bool):  # bool is an int subclass; not a measurement
            continue
        if isinstance(value, int | float):
            out.append(float(value))
    return out


def _descriptive_stats(values: list[float]) -> dict[str, float]:
    stats = {
        "count": float(len(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }
    if len(values) >= 2:
        stats["std"] = float(statistics.stdev(values))
    return stats
