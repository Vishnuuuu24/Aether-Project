"""FoundationEncoderBaselineEngine — docs/16 Sprint 10 DoD.

It must (1) satisfy the stable `BaselineEngine` protocol, (2) delegate the deviation
math to the statistical engine unchanged, (3) ingest raw PPG windows through the
learned extractor (with a classical fallback), and (4) abstain (None) when no HR can
be derived. Outputs carry THIS engine's version stamp.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from ai.baseline.config import BaselineConfig
from ai.baseline.foundation_encoder import (
    FOUNDATION_ENCODER_BASELINE_VERSION,
    FoundationEncoderBaselineEngine,
)
from ai.baseline.statistical import StatisticalBaselineEngine
from ai.features.foundation_encoder import FoundationEncoderFeatureExtractor
from ai.features.sqi import SqiGate
from ai.interfaces.baseline_engine import BaselineEngine
from schemas.baseline import BaselineAvailability, DeviationMagnitude
from schemas.features import RawWaveform, SignalWindow, WaveformKind
from schemas.reading import MeasurementContext, MetricCode, Reading

_PID = uuid4()
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_RATE = 64.0
_WIN = 512


def _config() -> BaselineConfig:
    # Session-scale sufficiency (like the WESAD eval): minutes not days, no circadian.
    return BaselineConfig(min_n=15, min_days=0, circadian_metrics=frozenset())


def _engine(*, weights: object | None = None) -> FoundationEncoderBaselineEngine:
    extractor = FoundationEncoderFeatureExtractor(weights)  # None => classical fallback
    delegate = StatisticalBaselineEngine(
        gate=SqiGate({"heart_rate": 0.0}),
        config=_config(),
        patient_id=_PID,
        version=FOUNDATION_ENCODER_BASELINE_VERSION,
    )
    return FoundationEncoderBaselineEngine(extractor=extractor, delegate=delegate)


def _tone_window(freq_hz: float, *, seq: int, n: int = _WIN) -> SignalWindow:
    ts = _BASE + timedelta(seconds=seq * 8)
    t = np.arange(n) / _RATE
    samples = np.sin(2 * np.pi * freq_hz * t).tolist()
    return SignalWindow(
        patient_id=_PID,
        metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING,
        window_start=ts,
        window_end=ts,
        waveform=RawWaveform(kind=WaveformKind.PPG, sample_rate_hz=_RATE, samples=samples),
    )


def _flat_window(*, seq: int) -> SignalWindow:
    ts = _BASE + timedelta(seconds=seq * 8)
    return SignalWindow(
        patient_id=_PID,
        metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING,
        window_start=ts,
        window_end=ts,
        waveform=RawWaveform(kind=WaveformKind.PPG, sample_rate_hz=_RATE, samples=[0.0] * _WIN),
    )


def _reading(value: float, *, seq: int) -> Reading:
    return Reading(
        patient_id=_PID,
        metric_code=MetricCode.HEART_RATE,
        value=value,
        unit="bpm",
        timestamp=_BASE + timedelta(hours=seq * 8),
        source_device="test",
        sqi=1.0,
        context=MeasurementContext.RESTING,
        ingest_adapter="test",
    )


def test_satisfies_baseline_engine_protocol() -> None:
    assert isinstance(_engine(), BaselineEngine)


def test_reading_path_delegates_and_stamps_engine_version() -> None:
    engine = _engine()
    for i in range(20):
        engine.update(_reading([58.0, 59.0, 60.0, 61.0, 62.0][i % 5], seq=i))
    baseline = engine.get_baseline("heart_rate", "resting")
    assert baseline.availability is BaselineAvailability.PERSONALISED
    assert baseline.baseline_engine_version == FOUNDATION_ENCODER_BASELINE_VERSION
    # A far-out reading flags a deviation (delegation to the statistical math is intact).
    result = engine.score(_reading(120.0, seq=99))
    assert result.magnitude is not DeviationMagnitude.NORMAL


# Slightly varied baseline rates (~76/78/80 bpm) so per-window HR has a real spread —
# a single fixed tone would give MAD sigma 0 and no deviation scale.
_BASELINE_FREQS = (1.27, 1.30, 1.33)


def test_window_path_derives_hr_and_personalises() -> None:
    engine = _engine()  # classical fallback extractor on clean ~78 bpm tones
    for i in range(20):
        reading = engine.update_from_window(_tone_window(_BASELINE_FREQS[i % 3], seq=i))
        assert reading is not None
        assert reading.metric_code is MetricCode.HEART_RATE
        assert reading.value > 0.0
    baseline = engine.get_baseline("heart_rate", "resting")
    assert baseline.availability is BaselineAvailability.PERSONALISED
    assert baseline.center is not None and 70.0 < baseline.center < 86.0


def test_window_path_scores_deviation() -> None:
    engine = _engine()
    for i in range(20):
        engine.update_from_window(_tone_window(_BASELINE_FREQS[i % 3], seq=i))  # ~78 bpm
    fast = engine.score_from_window(_tone_window(2.5, seq=99))  # ~150 bpm
    assert fast is not None
    assert fast.magnitude is not DeviationMagnitude.NORMAL


def test_window_path_abstains_when_no_hr() -> None:
    engine = _engine()
    assert engine.update_from_window(_flat_window(seq=0)) is None
    assert engine.score_from_window(_flat_window(seq=1)) is None


def test_from_checkpoint_missing_is_fallback_only(tmp_path: Path) -> None:
    engine = FoundationEncoderBaselineEngine.from_checkpoint(
        tmp_path / "nope",
        gate=SqiGate({"heart_rate": 0.0}),
        config=_config(),
        patient_id=_PID,
    )
    # Never raises; still derives HR via the classical fallback.
    reading = engine.update_from_window(_tone_window(1.3, seq=0))
    assert reading is not None and reading.value > 0.0
    assert engine.version == FOUNDATION_ENCODER_BASELINE_VERSION
