"""Contract tests for schemas/event.py."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.event import EventCandidate, EventStatus
from schemas.psg import EventSeverity


def _candidate(**overrides: object) -> EventCandidate:
    data: dict[str, object] = {
        "patient_id": uuid4(),
        "type": "physiological_stress/possible_illness",
        "severity": EventSeverity.MODERATE,
        "onset_ts": datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
        "contributing_deviation_ids": [uuid4()],
        "rule_id": "rule-1",
    }
    data.update(overrides)
    return EventCandidate(**data)  # type: ignore[arg-type]


def test_defaults_to_active() -> None:
    assert _candidate().status is EventStatus.ACTIVE


def test_rejects_naive_onset() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _candidate(onset_ts=datetime(2026, 6, 1, 8, 0))


def test_rejects_empty_contributing() -> None:
    with pytest.raises(ValidationError):
        _candidate(contributing_deviation_ids=[])
