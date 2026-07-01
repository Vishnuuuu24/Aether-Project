"""Per-reading SQI quality gate (docs/05 §3, §8).

Only readings with ``sqi >= threshold[metric]`` may enter the personal baseline
(``included_in_baseline=true``). Sub-threshold readings are kept but must never
pollute the baseline.

Thresholds are per-metric CLINICAL config (`config/clinical/sqi_thresholds.yaml`),
intentionally UNSET until a clinician provides them (CLAUDE.md: never fabricate
clinical thresholds). The gate is therefore **fail-safe**: a metric with no
configured threshold never passes, so nothing enters the baseline on an un-vetted
quality bar. Tests inject explicit thresholds to exercise the mechanism; production
ships the empty stub until clinical sign-off.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from schemas.reading import Reading


@dataclass(frozen=True)
class SqiGate:
    """Applies the per-metric SQI precondition. `thresholds` maps
    ``metric_code -> min sqi``; a metric absent from the map is unset (fail-safe).
    """

    thresholds: dict[str, float]

    def threshold_for(self, metric_code: str) -> float | None:
        return self.thresholds.get(metric_code)

    def passes(self, reading: Reading) -> bool:
        threshold = self.thresholds.get(reading.metric_code.value)
        if threshold is None:
            return False  # fail-safe: no clinical threshold => not quality-passing
        return reading.sqi >= threshold

    def apply(self, reading: Reading) -> Reading:
        """Return a copy with `included_in_baseline` set to the gate decision.

        Always overwrites the flag: a sender-supplied ``included_in_baseline`` is
        never trusted (mirrors the ingestion normaliser, docs/05 §8).
        """
        return reading.model_copy(update={"included_in_baseline": self.passes(reading)})

    def apply_batch(self, readings: Iterable[Reading]) -> list[Reading]:
        return [self.apply(reading) for reading in readings]
