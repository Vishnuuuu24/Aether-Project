"""FoundationEncoderBaselineEngine.from_papagei_checkpoint — docs/16 Sprint 10.

The PaPaGei-backed constructor must produce the SAME engine (protocol-conformant,
deviation math delegated unchanged, this-engine version stamped) — only the
HR-from-window extractor differs. A missing checkpoint falls back to classical inside
the extractor and must never raise.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from ai.baseline.foundation_encoder import (
    PAPAGEI_BASELINE_VERSION,
    FoundationEncoderBaselineEngine,
)
from ai.features.sqi import SqiGate
from ai.interfaces.baseline_engine import BaselineEngine
from schemas.features import RawWaveform, SignalWindow, WaveformKind
from schemas.reading import MeasurementContext, MetricCode

_PID = uuid4()
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_FS = 125.0
_WIN = 1250


def _ppg_window(freq_hz: float, *, seq: int) -> SignalWindow:
    ts = _BASE + timedelta(seconds=seq * 10)
    t = np.arange(_WIN) / _FS
    samples = np.sin(2 * np.pi * freq_hz * t).tolist()
    return SignalWindow(
        patient_id=_PID, metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING, window_start=ts, window_end=ts,
        waveform=RawWaveform(kind=WaveformKind.PPG, sample_rate_hz=_FS, samples=samples),
    )


def _engine(path: Path) -> FoundationEncoderBaselineEngine:
    return FoundationEncoderBaselineEngine.from_papagei_checkpoint(
        path, gate=SqiGate({"heart_rate": 0.0}), patient_id=_PID,
    )


def test_is_baseline_engine(tmp_path: Path) -> None:
    assert isinstance(_engine(tmp_path / "absent"), BaselineEngine)


def test_missing_checkpoint_never_raises_and_scores(tmp_path: Path) -> None:
    """Missing PaPaGei ckpt -> classical fallback in the extractor; the window path still
    derives HR and scores against the personal baseline (no exception)."""
    engine = _engine(tmp_path / "absent")
    # Build a personal baseline from several near-constant-HR windows, then score.
    for seq in range(20):
        engine.update_from_window(_ppg_window(1.25 + 0.01 * (seq % 3), seq=seq))
    result = engine.score_from_window(_ppg_window(2.5, seq=99))  # ~150 bpm-ish tone
    assert result is not None
    assert np.isfinite(result.z_robust)  # a real deviation score came back


def test_version_stamp(tmp_path: Path) -> None:
    engine = _engine(tmp_path / "absent")
    assert engine.version == PAPAGEI_BASELINE_VERSION
