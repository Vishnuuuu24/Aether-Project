"""EventEngine — T2.1 DoD (docs/10; docs/05 §6):
transient spikes suppressed; multi-metric events raised with contributing deviations.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from schemas.baseline import DeviationMagnitude
from schemas.event import EventStatus
from schemas.psg import DeviationDirection, DeviationNode, EventSeverity
from services.event_engine.engine import EventEngine
from services.event_engine.rules import CoOccurrenceRule, EventRuleSet, MetricCondition

_PID = uuid4()
_BASE = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)
_AS_OF = _BASE + timedelta(hours=1)


def _dev(
    metric: str,
    direction: DeviationDirection,
    z: float,
    *,
    minutes: int,
    confidence: float = 0.9,
    patient_id: UUID = _PID,
) -> DeviationNode:
    return DeviationNode(
        patient_id=patient_id,
        created_at=_BASE + timedelta(minutes=minutes),
        created_by="test",
        metric_code=metric,  # type: ignore[arg-type]
        baseline_id=uuid4(),
        magnitude=abs(z),
        direction=direction,
        z_robust=z,
        confidence=confidence,
        is_population_fallback=False,
    )


_STRESS_RULE = CoOccurrenceRule(
    id="stress-1",
    event_type="physiological_stress/possible_illness",
    window_minutes=120,
    persistence_count=2,
    conditions=(
        MetricCondition("heart_rate", DeviationDirection.UP, DeviationMagnitude.MODERATE),
        MetricCondition("respiratory_rate", DeviationDirection.UP, DeviationMagnitude.MILD),
    ),
)


def _engine(*rules: CoOccurrenceRule) -> EventEngine:
    return EventEngine(EventRuleSet(rules=rules, version="test-1"))


def test_multi_metric_event_raised_with_contributing_deviations() -> None:
    devs = [
        _dev("heart_rate", DeviationDirection.UP, 3.5, minutes=5),
        _dev("heart_rate", DeviationDirection.UP, 3.8, minutes=20),
        _dev("respiratory_rate", DeviationDirection.UP, 2.5, minutes=10),
        _dev("respiratory_rate", DeviationDirection.UP, 2.7, minutes=25),
    ]
    events = _engine(_STRESS_RULE).evaluate(_PID, devs, as_of=_AS_OF)

    assert len(events) == 1
    event = events[0]
    assert event.type == "physiological_stress/possible_illness"
    assert event.status is EventStatus.ACTIVE
    assert event.severity is EventSeverity.MODERATE  # worst |z|=3.8 -> moderate bucket
    assert set(event.contributing_deviation_ids) == {d.id for d in devs}
    assert event.onset_ts == _BASE + timedelta(minutes=5)  # earliest contributing


def test_transient_single_spike_suppressed() -> None:
    # Only ONE heart_rate deviation -> persistence_count=2 not met -> no event.
    devs = [
        _dev("heart_rate", DeviationDirection.UP, 3.5, minutes=5),
        _dev("respiratory_rate", DeviationDirection.UP, 2.5, minutes=10),
        _dev("respiratory_rate", DeviationDirection.UP, 2.7, minutes=25),
    ]
    assert _engine(_STRESS_RULE).evaluate(_PID, devs, as_of=_AS_OF) == []


def test_out_of_window_deviations_ignored() -> None:
    # HR deviations sit before the 120-min window -> excluded.
    devs = [
        _dev("heart_rate", DeviationDirection.UP, 3.5, minutes=-200),
        _dev("heart_rate", DeviationDirection.UP, 3.8, minutes=-180),
        _dev("respiratory_rate", DeviationDirection.UP, 2.5, minutes=10),
        _dev("respiratory_rate", DeviationDirection.UP, 2.7, minutes=25),
    ]
    assert _engine(_STRESS_RULE).evaluate(_PID, devs, as_of=_AS_OF) == []


def test_wrong_direction_or_magnitude_does_not_match() -> None:
    devs = [
        _dev("heart_rate", DeviationDirection.DOWN, 3.5, minutes=5),  # wrong direction
        _dev("heart_rate", DeviationDirection.UP, 2.1, minutes=6),  # only mild < moderate
        _dev("respiratory_rate", DeviationDirection.UP, 2.5, minutes=10),
        _dev("respiratory_rate", DeviationDirection.UP, 2.7, minutes=25),
    ]
    assert _engine(_STRESS_RULE).evaluate(_PID, devs, as_of=_AS_OF) == []


def test_acute_red_flag_bypasses_persistence() -> None:
    red_flag = CoOccurrenceRule(
        id="acute-1",
        event_type="acute_flag",
        window_minutes=60,
        persistence_count=5,  # ignored for red-flags
        conditions=(
            MetricCondition("heart_rate", DeviationDirection.UP, DeviationMagnitude.MARKED),
        ),
        acute_red_flag=True,
    )
    devs = [_dev("heart_rate", DeviationDirection.UP, 5.0, minutes=5)]  # single reading
    events = _engine(red_flag).evaluate(_PID, devs, as_of=_AS_OF)
    assert len(events) == 1
    assert events[0].severity is EventSeverity.HIGH  # marked bucket


def test_low_confidence_downgrades_severity() -> None:
    devs = [
        _dev("heart_rate", DeviationDirection.UP, 3.5, minutes=5, confidence=0.2),
        _dev("heart_rate", DeviationDirection.UP, 3.8, minutes=20, confidence=0.2),
        _dev("respiratory_rate", DeviationDirection.UP, 2.5, minutes=10, confidence=0.2),
        _dev("respiratory_rate", DeviationDirection.UP, 2.7, minutes=25, confidence=0.2),
    ]
    events = _engine(_STRESS_RULE).evaluate(_PID, devs, as_of=_AS_OF)
    assert events[0].severity is EventSeverity.LOW  # moderate downgraded on low confidence


def test_empty_ruleset_raises_nothing() -> None:
    devs = [_dev("heart_rate", DeviationDirection.UP, 5.0, minutes=5)]
    assert EventEngine(EventRuleSet(rules=())).evaluate(_PID, devs, as_of=_AS_OF) == []


def test_other_patients_deviations_excluded() -> None:
    other = uuid4()
    devs = [
        _dev("heart_rate", DeviationDirection.UP, 3.5, minutes=5, patient_id=other),
        _dev("heart_rate", DeviationDirection.UP, 3.8, minutes=20, patient_id=other),
        _dev("respiratory_rate", DeviationDirection.UP, 2.5, minutes=10),
        _dev("respiratory_rate", DeviationDirection.UP, 2.7, minutes=25),
    ]
    assert _engine(_STRESS_RULE).evaluate(_PID, devs, as_of=_AS_OF) == []


def test_naive_as_of_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _engine(_STRESS_RULE).evaluate(_PID, [], as_of=datetime(2026, 6, 1, 9, 0))
