"""SqiGate: per-metric quality gate + fail-safe on unset thresholds (docs/05 §3, §8)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ai.features.sqi import SqiGate
from schemas.reading import MeasurementContext, MetricCode, Reading


def _reading(**overrides: Any) -> Reading:
    data: dict[str, Any] = {
        "patient_id": uuid4(),
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


def test_passes_when_at_or_above_threshold() -> None:
    gate = SqiGate({"heart_rate": 0.8})
    assert gate.passes(_reading(sqi=0.9)) is True
    assert gate.passes(_reading(sqi=0.8)) is True  # boundary is inclusive


def test_fails_below_threshold() -> None:
    gate = SqiGate({"heart_rate": 0.8})
    assert gate.passes(_reading(sqi=0.79)) is False


def test_unset_metric_is_fail_safe() -> None:
    # Empty config (the shipped stub) => nothing passes, regardless of sqi.
    gate = SqiGate({})
    assert gate.passes(_reading(sqi=1.0)) is False
    assert gate.threshold_for("heart_rate") is None


def test_threshold_applies_per_metric() -> None:
    gate = SqiGate({"heart_rate": 0.8})  # spo2 unset
    assert gate.passes(_reading(metric_code=MetricCode.HEART_RATE, sqi=0.85)) is True
    assert gate.passes(_reading(metric_code=MetricCode.SPO2, sqi=0.99)) is False


def test_apply_sets_included_in_baseline() -> None:
    gate = SqiGate({"heart_rate": 0.8})
    passed = gate.apply(_reading(sqi=0.9, included_in_baseline=False))
    failed = gate.apply(_reading(sqi=0.5, included_in_baseline=False))
    assert passed.included_in_baseline is True
    assert failed.included_in_baseline is False


def test_apply_overwrites_sender_supplied_flag() -> None:
    # A sub-threshold reading claiming included_in_baseline=True is corrected to False.
    gate = SqiGate({"heart_rate": 0.8})
    corrected = gate.apply(_reading(sqi=0.5, included_in_baseline=True))
    assert corrected.included_in_baseline is False


def test_apply_batch() -> None:
    gate = SqiGate({"heart_rate": 0.8})
    out = gate.apply_batch([_reading(sqi=0.9), _reading(sqi=0.4), _reading(sqi=0.81)])
    assert [r.included_in_baseline for r in out] == [True, False, True]
