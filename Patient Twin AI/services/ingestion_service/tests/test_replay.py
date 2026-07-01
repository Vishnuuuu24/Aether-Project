"""T1.1 DoD: replay of PPG-DaLiA produces normalised readings.

Uses a small in-memory record in the PPG-DaLiA shape (lists, not numpy) so the
stream→normalise pipeline is proven without downloading the dataset.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from schemas.reading import MeasurementContext, MetricCode
from services.ingestion_service.adapters.replay import (
    ADAPTER_NAME,
    SOURCE_DEVICE,
    stream_ppg_dalia,
)
from services.ingestion_service.normaliser import normalise_batch


def sample() -> dict[str, Any]:
    # HR ground truth @0.5 Hz; activity @4 Hz; wrist TEMP @4 Hz.
    return {
        "label": [70.0, 72.0, 95.0, 110.0],
        "activity": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4],
        "signal": {"wrist": {"TEMP": [33.0, 33.1, 33.2]}},
        "subject": "S-test",
    }


def test_replay_produces_normalised_readings() -> None:
    pid = uuid4()
    raws = list(
        stream_ppg_dalia(sample(), patient_id=pid, base_ts=datetime(2026, 1, 1, tzinfo=UTC))
    )
    assert len(raws) == 7  # 4 heart_rate + 3 skin_temp

    result = normalise_batch(raws, default_adapter=ADAPTER_NAME)
    assert result.rejections == []
    readings = [r for _, r in result.accepted]
    assert len(readings) == 7

    hr = [r for r in readings if r.metric_code is MetricCode.HEART_RATE]
    temp = [r for r in readings if r.metric_code is MetricCode.SKIN_TEMP]
    assert len(hr) == 4
    assert len(temp) == 3
    assert all(r.source_device == SOURCE_DEVICE for r in readings)
    assert all(r.timestamp.tzinfo is not None for r in readings)
    assert all(r.patient_id == pid for r in readings)
    assert all(r.ingest_adapter == ADAPTER_NAME for r in readings)


def test_replay_maps_activity_to_context() -> None:
    raws = list(stream_ppg_dalia(sample(), patient_id=uuid4()))
    hr = [r for r in raws if r["metric_code"] == MetricCode.HEART_RATE.value]
    # index 0 → t=0s → activity[0]=1 (sitting) → resting
    assert hr[0]["context"] == MeasurementContext.RESTING.value
    # index 1 → t=2s → activity[8]=2 (stairs) → active
    assert hr[1]["context"] == MeasurementContext.ACTIVE.value


def test_replay_empty_record_yields_nothing() -> None:
    assert list(stream_ppg_dalia({}, patient_id=uuid4())) == []
