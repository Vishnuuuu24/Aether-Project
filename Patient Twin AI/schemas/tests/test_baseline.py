"""Contract tests for schemas/baseline.py."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.baseline import (
    Baseline,
    BaselineAvailability,
    DeviationMagnitude,
    DeviationResult,
    PopulationRange,
)
from schemas.psg import DeviationDirection
from schemas.reading import MeasurementContext, MetricCode


def _baseline(**overrides: object) -> Baseline:
    data: dict[str, object] = {
        "patient_id": uuid4(),
        "metric_code": MetricCode.HEART_RATE,
        "context": MeasurementContext.RESTING,
        "availability": BaselineAvailability.PERSONALISED,
        "center": 60.0,
        "dispersion_sigma": 1.48,
        "sample_n": 60,
        "span_days": 20.0,
        "window_days": 28,
        "min_n": 50,
        "min_days": 7,
        "is_population_fallback": False,
        "baseline_engine_version": "statistical-v1",
    }
    data.update(overrides)
    return Baseline(**data)  # type: ignore[arg-type]


def test_population_range_rejects_inverted() -> None:
    with pytest.raises(ValidationError, match="high must be >= low"):
        PopulationRange(low=100.0, high=60.0, unit="bpm")


def test_fallback_flag_must_track_availability() -> None:
    with pytest.raises(ValidationError, match="is_population_fallback"):
        _baseline(availability=BaselineAvailability.PERSONALISED, is_population_fallback=True)


def test_personalised_requires_center_and_sigma() -> None:
    with pytest.raises(ValidationError, match="center and dispersion_sigma"):
        _baseline(center=None)


def test_unavailable_baseline_allows_null_center() -> None:
    b = _baseline(
        availability=BaselineAvailability.UNAVAILABLE,
        center=None,
        dispersion_sigma=None,
        is_population_fallback=False,
        sample_n=0,
    )
    assert b.center is None


def test_deviation_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        DeviationResult(
            reading_id=uuid4(),
            patient_id=uuid4(),
            metric_code=MetricCode.HEART_RATE,
            context=MeasurementContext.RESTING,
            z_robust=1.0,
            direction=DeviationDirection.UP,
            magnitude=DeviationMagnitude.NORMAL,
            confidence=1.5,  # out of [0, 1]
            is_population_fallback=False,
            baseline_availability=BaselineAvailability.PERSONALISED,
        )


def test_deviation_defaults_uncalibrated() -> None:
    dev = DeviationResult(
        reading_id=uuid4(),
        patient_id=uuid4(),
        metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING,
        z_robust=0.0,
        direction=DeviationDirection.NONE,
        magnitude=DeviationMagnitude.NORMAL,
        confidence=0.0,
        is_population_fallback=False,
        baseline_availability=BaselineAvailability.UNAVAILABLE,
    )
    assert dev.confidence_calibrated is False
    assert dev.baseline_ref is None
