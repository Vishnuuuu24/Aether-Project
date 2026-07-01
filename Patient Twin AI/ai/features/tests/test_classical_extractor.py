"""ClassicalFeatureExtractor — T1.2 DoD:
sub-threshold readings excluded from the baseline; features computed for
required-core metrics (docs/10 T1.2; docs/05 §3-4).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from ai.features.classical import FEATURE_EXTRACTOR_VERSION, ClassicalFeatureExtractor
from ai.features.sqi import SqiGate
from schemas.features import SignalWindow
from schemas.reading import MeasurementContext, MetricCode, Reading

_PID = uuid4()
_BASE = datetime(2026, 6, 1, tzinfo=UTC)


def _reading(i: int, value: float, sqi: float, **overrides: Any) -> Reading:
    data: dict[str, Any] = {
        "patient_id": _PID,
        "metric_code": MetricCode.HEART_RATE,
        "value": value,
        "unit": "bpm",
        "timestamp": _BASE + timedelta(hours=i),
        "source_device": "apple_watch_s9",
        "sqi": sqi,
        "context": MeasurementContext.RESTING,
        "ingest_adapter": "csv",
    }
    data.update(overrides)
    return Reading(**data)


def _window(readings: list[Reading], metric: MetricCode = MetricCode.HEART_RATE) -> SignalWindow:
    return SignalWindow(
        patient_id=_PID,
        metric_code=metric,
        context=MeasurementContext.RESTING,
        window_start=_BASE,
        window_end=_BASE + timedelta(days=1),
        readings=readings,
    )


def test_features_computed_over_quality_passing_only() -> None:
    extractor = ClassicalFeatureExtractor(SqiGate({"heart_rate": 0.8}))
    window = _window(
        [
            _reading(0, 60.0, sqi=0.9),
            _reading(1, 62.0, sqi=0.85),
            _reading(2, 200.0, sqi=0.2),  # artefact: sub-threshold, must be excluded
        ]
    )
    fs = extractor.extract(window)
    assert fs.n_total == 3
    assert fs.n_quality_passing == 2
    assert fs.features["count"] == 2.0
    assert fs.features["mean"] == 61.0  # 200.0 artefact did not leak in
    assert fs.features["max"] == 62.0
    assert fs.sqi_threshold_applied == 0.8
    assert fs.feature_extractor_version == FEATURE_EXTRACTOR_VERSION


def test_required_core_metric_steps_features_computed() -> None:
    extractor = ClassicalFeatureExtractor(SqiGate({"steps": 0.5}))
    window = _window(
        [
            _reading(0, 1000.0, sqi=0.9, metric_code=MetricCode.STEPS, unit="count"),
            _reading(1, 1500.0, sqi=0.9, metric_code=MetricCode.STEPS, unit="count"),
        ],
        metric=MetricCode.STEPS,
    )
    fs = extractor.extract(window)
    assert fs.n_quality_passing == 2
    assert fs.features["mean"] == 1250.0
    assert fs.features["std"] > 0.0


def test_unset_threshold_yields_no_passing_and_no_features() -> None:
    # Shipped stub state: no clinical threshold => nothing enters the baseline.
    extractor = ClassicalFeatureExtractor(SqiGate({}))
    fs = extractor.extract(_window([_reading(0, 60.0, sqi=1.0)]))
    assert fs.n_total == 1
    assert fs.n_quality_passing == 0
    assert fs.features == {}
    assert fs.sqi_threshold_applied is None


def test_single_passing_reading_has_no_std() -> None:
    extractor = ClassicalFeatureExtractor(SqiGate({"heart_rate": 0.8}))
    fs = extractor.extract(_window([_reading(0, 60.0, sqi=0.9)]))
    assert fs.features["count"] == 1.0
    assert "std" not in fs.features


def test_empty_window() -> None:
    extractor = ClassicalFeatureExtractor(SqiGate({"heart_rate": 0.8}))
    fs = extractor.extract(_window([]))
    assert fs.n_total == 0
    assert fs.n_quality_passing == 0
    assert fs.features == {}
