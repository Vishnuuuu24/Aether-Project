"""Per-reading validation (docs/04 §2). The headline rule: a reading without a
timezone is rejected.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas import MeasurementContext, MetricCode, Reading


def base(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "patient_id": uuid4(),
        "metric_code": MetricCode.HEART_RATE,
        "value": 58,
        "unit": "bpm",
        "timestamp": datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
        "source_device": "apple_watch_s9",
        "sqi": 0.95,
        "context": MeasurementContext.RESTING,
        "ingest_adapter": "healthkit",
    }
    data.update(overrides)
    return data


def test_valid_scalar_reading() -> None:
    r = Reading(**base())
    assert r.value == 58
    assert r.included_in_baseline is False  # SQI gate sets this, not the sender


def test_structured_value_allowed() -> None:
    r = Reading(
        **base(metric_code=MetricCode.SLEEP, value={"deep_min": 90, "rem_min": 60}, unit="min")
    )
    assert isinstance(r.value, dict)


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        Reading(**base(timestamp=datetime(2026, 6, 1, 7, 30)))  # no tzinfo


def test_blank_unit_rejected() -> None:
    with pytest.raises(ValidationError):
        Reading(**base(unit="   "))


def test_sqi_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Reading(**base(sqi=1.5))
    with pytest.raises(ValidationError):
        Reading(**base(sqi=-0.1))


def test_roundtrip_serialise_validate() -> None:
    r = Reading(**base(raw_ref="s3://raw/window/abc"))
    assert Reading.model_validate_json(r.model_dump_json()) == r
