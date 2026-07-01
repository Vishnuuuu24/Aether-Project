"""StatisticalBaselineEngine — T1.3 DoD (docs/10; docs/05 §3-5, §8):
baselines personalise only after sufficiency; population fallback flagged;
injected artefacts don't move the baseline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from ai.baseline.config import BaselineConfig
from ai.baseline.population import StaticPopulationReferenceProvider
from ai.baseline.statistical import StatisticalBaselineEngine
from ai.features.sqi import SqiGate
from schemas.baseline import BaselineAvailability, DeviationMagnitude, PopulationRange
from schemas.psg import DeviationDirection
from schemas.reading import MeasurementContext, MetricCode, Reading

_PID = uuid4()
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_THRESHOLD = {"heart_rate": 0.5}
# Symmetric spread around 60 => median 60, MAD 1 => robust sigma = 1.4826.
_CYCLE = [58.0, 59.0, 60.0, 61.0, 62.0]


def _reading(value: float, *, hours: float, sqi: float = 0.9, **overrides: Any) -> Reading:
    data: dict[str, Any] = {
        "patient_id": _PID,
        "metric_code": MetricCode.HEART_RATE,
        "value": value,
        "unit": "bpm",
        "timestamp": _BASE + timedelta(hours=hours),
        "source_device": "apple_watch_s9",
        "sqi": sqi,
        "context": MeasurementContext.RESTING,
        "ingest_adapter": "csv",
    }
    data.update(overrides)
    return Reading(**data)


def _engine(**kwargs: Any) -> StatisticalBaselineEngine:
    kwargs.setdefault("gate", SqiGate(_THRESHOLD))
    kwargs.setdefault("patient_id", _PID)
    return StatisticalBaselineEngine(**kwargs)


def _feed_personalising(engine: StatisticalBaselineEngine, *, n: int = 60) -> None:
    """n quality-passing readings over ~20 days (within the 28d window)."""
    for i in range(n):
        engine.update(_reading(_CYCLE[i % len(_CYCLE)], hours=i * 8))


# -- sufficiency ------------------------------------------------------------


def test_below_sufficiency_is_not_personalised() -> None:
    engine = _engine()  # no population provider
    for i in range(10):  # < min_n
        engine.update(_reading(_CYCLE[i % len(_CYCLE)], hours=i * 8))
    baseline = engine.get_baseline("heart_rate", "resting")
    assert baseline.availability is BaselineAvailability.UNAVAILABLE
    assert baseline.is_population_fallback is False


def test_personalises_after_sufficiency() -> None:
    engine = _engine()
    _feed_personalising(engine)
    baseline = engine.get_baseline("heart_rate", "resting")
    assert baseline.availability is BaselineAvailability.PERSONALISED
    assert baseline.is_population_fallback is False
    assert baseline.center == 60.0
    assert baseline.dispersion_sigma == pytest.approx(1.4826)
    assert baseline.sample_n == 60
    assert baseline.ewma is not None


def test_enough_readings_but_too_short_span_not_personalised() -> None:
    engine = _engine()
    for i in range(60):  # 60 readings but packed into a few hours => span < min_days
        engine.update(_reading(_CYCLE[i % len(_CYCLE)], hours=i * 0.01))
    assert engine.get_baseline("heart_rate", "resting").availability is not (
        BaselineAvailability.PERSONALISED
    )


# -- population fallback ----------------------------------------------------


def test_population_fallback_flagged_when_insufficient() -> None:
    provider = StaticPopulationReferenceProvider(
        {"heart_rate": PopulationRange(low=60.0, high=100.0, unit="bpm")}
    )
    engine = _engine(population_provider=provider)
    engine.update(_reading(70.0, hours=0))  # < min_n
    baseline = engine.get_baseline("heart_rate", "resting")
    assert baseline.availability is BaselineAvailability.POPULATION_FALLBACK
    assert baseline.is_population_fallback is True
    assert baseline.center == 80.0  # (60 + 100) / 2
    assert baseline.dispersion_sigma == pytest.approx(10.0)  # (100 - 60) / 4


def test_transition_fallback_to_personalised_flips_flag() -> None:
    provider = StaticPopulationReferenceProvider(
        {"heart_rate": PopulationRange(low=60.0, high=100.0, unit="bpm")}
    )
    engine = _engine(population_provider=provider)
    engine.update(_reading(60.0, hours=0))
    assert engine.get_baseline("heart_rate", "resting").is_population_fallback is True
    _feed_personalising(engine)
    after = engine.get_baseline("heart_rate", "resting")
    assert after.availability is BaselineAvailability.PERSONALISED
    assert after.is_population_fallback is False


# -- robustness (DoD): artefacts must not move the baseline ------------------


def test_injected_artefacts_do_not_move_baseline() -> None:
    engine = _engine()
    _feed_personalising(engine)
    before = engine.get_baseline("heart_rate", "resting")
    # 40 wild high-value readings, all sub-threshold (sqi below the gate).
    for i in range(40):
        engine.update(_reading(200.0, hours=1000 + i, sqi=0.1))
    after = engine.get_baseline("heart_rate", "resting")
    assert after.center == before.center
    assert after.dispersion_sigma == before.dispersion_sigma
    assert after.sample_n == before.sample_n


# -- deviation scoring ------------------------------------------------------


def test_scores_marked_deviation() -> None:
    engine = _engine()
    _feed_personalising(engine)
    dev = engine.score(_reading(80.0, hours=2000))  # z = (80-60)/1.4826 ≈ 13.5
    assert dev.direction is DeviationDirection.UP
    assert dev.magnitude is DeviationMagnitude.MARKED
    assert dev.z_robust > 4.5
    assert dev.is_population_fallback is False
    assert dev.confidence_calibrated is False
    assert dev.confidence > 0.0


def test_scores_normal_at_center() -> None:
    engine = _engine()
    _feed_personalising(engine)
    dev = engine.score(_reading(60.0, hours=2000))
    assert dev.direction is DeviationDirection.NONE
    assert dev.magnitude is DeviationMagnitude.NORMAL
    assert dev.z_robust == 0.0


def test_scores_mild_deviation() -> None:
    engine = _engine()
    _feed_personalising(engine)
    dev = engine.score(_reading(63.0, hours=2000))  # z = 3/1.4826 ≈ 2.02
    assert dev.magnitude is DeviationMagnitude.MILD
    assert dev.direction is DeviationDirection.UP


def test_score_unavailable_when_no_basis() -> None:
    engine = _engine()  # no readings, no provider
    dev = engine.score(_reading(200.0, hours=0))
    assert dev.baseline_availability is BaselineAvailability.UNAVAILABLE
    assert dev.magnitude is DeviationMagnitude.NORMAL
    assert dev.confidence == 0.0
    assert dev.z_robust == 0.0


def test_population_fallback_scoring_penalises_confidence() -> None:
    provider = StaticPopulationReferenceProvider(
        {"heart_rate": PopulationRange(low=60.0, high=100.0, unit="bpm")}
    )
    engine = _engine(population_provider=provider)
    engine.update(_reading(80.0, hours=0))
    dev = engine.score(_reading(110.0, hours=1))  # z = (110-80)/10 = 3.0
    assert dev.is_population_fallback is True
    assert dev.magnitude is DeviationMagnitude.MODERATE
    assert dev.confidence <= 0.5  # fallback penalty applied


# -- circadian stratification ----------------------------------------------


def test_circadian_bucket_used_when_it_has_sufficiency() -> None:
    # Small config so a single time-of-day bucket can personalise on its own.
    cfg = BaselineConfig(min_n=5, min_days=0)
    engine = _engine(config=cfg)
    # 6 "morning" readings centred ~50, 6 "night" readings centred ~90.
    for i in range(6):
        engine.update(_reading(48.0 + (i % 3), hours=8 + i * 24))  # 08:00 => morning
    for i in range(6):
        engine.update(_reading(88.0 + (i % 3), hours=2 + i * 24))  # 02:00 => night
    # A morning reading is scored against the morning (~50) baseline, not the pooled mean.
    dev = engine.score(_reading(50.0, hours=8 + 6 * 24))
    assert dev.baseline_availability is BaselineAvailability.PERSONALISED
    assert dev.magnitude is DeviationMagnitude.NORMAL


# -- per-patient scoping ----------------------------------------------------


def test_rejects_foreign_patient() -> None:
    engine = _engine()
    other = _reading(60.0, hours=0)
    foreign = other.model_copy(update={"patient_id": UUID(int=other.patient_id.int ^ 1)})
    with pytest.raises(ValueError, match="per-patient"):
        engine.update(foreign)
