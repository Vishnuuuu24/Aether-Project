"""EventEngine — combine deviations into candidate events (docs/05 §6).

For each versioned co-occurrence rule, look at the deviations in the rule's trailing
window and require EACH condition to be met by at least `persistence_count` matching
deviations (transient single-reading spikes are suppressed) — unless the rule is a
configured acute red-flag, which a single reading may raise. When every condition is
met, emit an `EventCandidate` whose `contributing_deviation_ids` are the matching
deviations, with a deterministically derived severity.

Pure and stateless: it reads deviations and returns candidates. Committing them to
the PSG (append-only + audited) is the Patient State Engine's job (docs/02 §4).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from uuid import UUID

from schemas.baseline import MAGNITUDE_RANK, magnitude_bucket
from schemas.event import EventCandidate, EventStatus
from schemas.psg import DeviationNode, EventSeverity

from .rules import CoOccurrenceRule, EventRuleSet, MetricCondition

EVENT_ENGINE_VERSION = "event-engine-v1"

# Worst contributing magnitude bucket -> base severity (structural, docs/05 §6).
_BUCKET_SEVERITY: dict[int, EventSeverity] = {
    0: EventSeverity.NONE,  # normal
    1: EventSeverity.LOW,  # mild
    2: EventSeverity.MODERATE,  # moderate
    3: EventSeverity.HIGH,  # marked
}
_SEVERITY_ORDER = [
    EventSeverity.NONE,
    EventSeverity.LOW,
    EventSeverity.MODERATE,
    EventSeverity.HIGH,
]
_LOW_CONFIDENCE = 0.5  # below this, severity is downgraded one level (uncalibrated, docs/05 §5)


class EventEngine:
    def __init__(self, ruleset: EventRuleSet, *, version: str = EVENT_ENGINE_VERSION) -> None:
        self._ruleset = ruleset
        self._version = version

    def evaluate(
        self, patient_id: UUID, deviations: Sequence[DeviationNode], *, as_of: datetime
    ) -> list[EventCandidate]:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        mine = [d for d in deviations if d.patient_id == patient_id]
        candidates: list[EventCandidate] = []
        for rule in self._ruleset.rules:
            candidate = self._apply_rule(patient_id, rule, mine, as_of)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _apply_rule(
        self,
        patient_id: UUID,
        rule: CoOccurrenceRule,
        deviations: Sequence[DeviationNode],
        as_of: datetime,
    ) -> EventCandidate | None:
        window_start = as_of - timedelta(minutes=rule.window_minutes)
        in_window = [d for d in deviations if window_start <= d.created_at <= as_of]
        required = 1 if rule.acute_red_flag else max(1, rule.persistence_count)

        contributing: list[DeviationNode] = []
        for condition in rule.conditions:
            matches = [d for d in in_window if _matches(d, condition)]
            if len(matches) < required:
                return None  # condition not persistently met — suppress
            contributing.extend(matches)

        ids = _unique_ids(contributing)
        onset = min(d.created_at for d in contributing)
        return EventCandidate(
            patient_id=patient_id,
            type=rule.event_type,
            severity=_severity(contributing),
            status=EventStatus.ACTIVE,
            onset_ts=onset,
            contributing_deviation_ids=ids,
            rule_id=rule.id,
        )


def _matches(deviation: DeviationNode, condition: MetricCondition) -> bool:
    if deviation.metric_code.value != condition.metric_code:
        return False
    if deviation.direction != condition.direction:
        return False
    bucket = magnitude_bucket(abs(deviation.z_robust))
    return MAGNITUDE_RANK[bucket] >= MAGNITUDE_RANK[condition.min_magnitude]


def _unique_ids(deviations: Sequence[DeviationNode]) -> list[UUID]:
    seen: set[UUID] = set()
    ordered: list[UUID] = []
    for d in deviations:
        if d.id not in seen:
            seen.add(d.id)
            ordered.append(d.id)
    return ordered


def _severity(contributing: Sequence[DeviationNode]) -> EventSeverity:
    """Severity from the worst contributing deviation, downgraded when confidence is
    low (docs/05 §6). Deterministic and structural — not a clinical threshold.
    """
    worst = max(contributing, key=lambda d: abs(d.z_robust))
    bucket = magnitude_bucket(abs(worst.z_robust))
    severity = _BUCKET_SEVERITY[MAGNITUDE_RANK[bucket]]
    mean_confidence = sum(d.confidence for d in contributing) / len(contributing)
    if mean_confidence < _LOW_CONFIDENCE:
        idx = max(0, _SEVERITY_ORDER.index(severity) - 1)
        severity = _SEVERITY_ORDER[idx]
    return severity
