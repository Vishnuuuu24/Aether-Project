"""Contract tests for SignalWindow / FeatureSet (schemas/features.py)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.features import FeatureSet, SignalWindow
from schemas.reading import MeasurementContext, MetricCode, Reading


def _reading(patient_id: Any, **overrides: Any) -> Reading:
    data: dict[str, Any] = {
        "patient_id": patient_id,
        "metric_code": MetricCode.HEART_RATE,
        "value": 58.0,
        "unit": "bpm",
        "timestamp": datetime(2026, 6, 1, 7, 30, tzinfo=UTC),
        "source_device": "apple_watch_s9",
        "sqi": 0.9,
        "context": MeasurementContext.RESTING,
        "ingest_adapter": "csv",
    }
    data.update(overrides)
    return Reading(**data)


def _window(**overrides: Any) -> SignalWindow:
    pid = overrides.pop("patient_id", uuid4())
    data: dict[str, Any] = {
        "patient_id": pid,
        "metric_code": MetricCode.HEART_RATE,
        "context": MeasurementContext.RESTING,
        "window_start": datetime(2026, 6, 1, tzinfo=UTC),
        "window_end": datetime(2026, 6, 8, tzinfo=UTC),
        "readings": [],
    }
    data.update(overrides)
    return SignalWindow(**data)


def test_window_accepts_matching_readings() -> None:
    pid = uuid4()
    window = _window(patient_id=pid, readings=[_reading(pid), _reading(pid, value=61.0)])
    assert len(window.readings) == 2


def test_window_rejects_reading_from_other_patient() -> None:
    pid = uuid4()
    with pytest.raises(ValidationError, match="patient_id"):
        _window(patient_id=pid, readings=[_reading(uuid4())])


def test_window_rejects_metric_mismatch() -> None:
    pid = uuid4()
    with pytest.raises(ValidationError, match="metric_code"):
        _window(patient_id=pid, readings=[_reading(pid, metric_code=MetricCode.SPO2)])


def test_window_rejects_context_mismatch() -> None:
    pid = uuid4()
    with pytest.raises(ValidationError, match="context"):
        _window(patient_id=pid, readings=[_reading(pid, context=MeasurementContext.ACTIVE)])


def test_window_rejects_naive_bounds() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _window(window_start=datetime(2026, 6, 1))


def test_window_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="window_end"):
        _window(
            window_start=datetime(2026, 6, 8, tzinfo=UTC),
            window_end=datetime(2026, 6, 1, tzinfo=UTC),
        )


def test_featureset_rejects_passing_over_total() -> None:
    with pytest.raises(ValidationError, match="n_quality_passing"):
        FeatureSet(
            patient_id=uuid4(),
            metric_code=MetricCode.HEART_RATE,
            context=MeasurementContext.RESTING,
            window_start=datetime(2026, 6, 1, tzinfo=UTC),
            window_end=datetime(2026, 6, 8, tzinfo=UTC),
            n_total=1,
            n_quality_passing=2,
            feature_extractor_version="classical-v1",
        )


def test_featureset_defaults_threshold_none() -> None:
    fs = FeatureSet(
        patient_id=uuid4(),
        metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING,
        window_start=datetime(2026, 6, 1, tzinfo=UTC),
        window_end=datetime(2026, 6, 8, tzinfo=UTC),
        n_total=0,
        n_quality_passing=0,
        feature_extractor_version="classical-v1",
    )
    assert fs.sqi_threshold_applied is None
    assert fs.features == {}
