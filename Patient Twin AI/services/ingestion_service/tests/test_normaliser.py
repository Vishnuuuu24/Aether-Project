"""T1.1 DoD: readings missing required metadata are rejected with field errors."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

from services.ingestion_service.normaliser import (
    UNKNOWN_SQI,
    ReadingRejected,
    normalise_batch,
    normalise_one,
)


def raw(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "patient_id": str(uuid4()),
        "metric_code": "heart_rate",
        "value": 58,
        "unit": "bpm",
        "timestamp": datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
        "source_device": "apple_watch_s9",
        "context": "resting",
    }
    data.update(overrides)
    return data


def rejected_fields(exc: ReadingRejected) -> set[str]:
    return {e.field for e in exc.errors}


def test_valid_reading_normalises_with_defaults() -> None:
    reading = normalise_one(raw(), default_adapter="csv")
    assert reading.sqi == UNKNOWN_SQI  # unknown until the SQI service scores it
    assert reading.included_in_baseline is False
    assert reading.ingest_adapter == "csv"


def test_sender_cannot_set_included_in_baseline() -> None:
    reading = normalise_one(raw(included_in_baseline=True), default_adapter="csv")
    assert reading.included_in_baseline is False


def test_explicit_ingest_adapter_preserved() -> None:
    reading = normalise_one(raw(ingest_adapter="fitbit"), default_adapter="csv")
    assert reading.ingest_adapter == "fitbit"


@pytest.mark.parametrize("missing", ["metric_code", "value", "unit", "source_device", "context"])
def test_missing_required_field_rejected(missing: str) -> None:
    data = raw()
    del data[missing]
    with pytest.raises(ReadingRejected) as exc:
        normalise_one(data, default_adapter="csv")
    assert missing in rejected_fields(exc.value)


def test_missing_timestamp_rejected() -> None:
    data = raw()
    del data["timestamp"]
    with pytest.raises(ReadingRejected) as exc:
        normalise_one(data, default_adapter="csv")
    assert "timestamp" in rejected_fields(exc.value)


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ReadingRejected) as exc:
        normalise_one(raw(timestamp=datetime(2026, 6, 1, 7, 30)), default_adapter="csv")
    assert "timestamp" in rejected_fields(exc.value)


def test_blank_unit_rejected() -> None:
    with pytest.raises(ReadingRejected) as exc:
        normalise_one(raw(unit="   "), default_adapter="csv")
    assert "unit" in rejected_fields(exc.value)


def test_invalid_metric_code_rejected() -> None:
    with pytest.raises(ReadingRejected) as exc:
        normalise_one(raw(metric_code="blood_unicorn"), default_adapter="csv")
    assert "metric_code" in rejected_fields(exc.value)


def test_sqi_out_of_range_rejected() -> None:
    with pytest.raises(ReadingRejected) as exc:
        normalise_one(raw(sqi=1.7), default_adapter="csv")
    assert "sqi" in rejected_fields(exc.value)


def test_batch_keeps_accepted_and_rejected_with_indices() -> None:
    good = raw()
    bad = raw()
    del bad["unit"]
    result = normalise_batch([good, bad, raw()], default_adapter="csv")
    assert [i for i, _ in result.accepted] == [0, 2]
    assert len(result.rejections) == 1
    assert result.rejections[0]["index"] == 1
