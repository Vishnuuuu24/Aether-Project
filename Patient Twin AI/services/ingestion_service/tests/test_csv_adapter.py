"""CSV adapter → shared normaliser."""

from __future__ import annotations

from uuid import uuid4

from services.ingestion_service.adapters.csv_adapter import ADAPTER_NAME, rows_from_text
from services.ingestion_service.normaliser import normalise_batch


def test_csv_rows_normalise() -> None:
    p1, p2 = uuid4(), uuid4()
    text = (
        "patient_id,metric_code,value,unit,timestamp,source_device,context\n"
        f"{p1},heart_rate,58,bpm,2026-06-01T07:30:00+00:00,apple_watch_s9,resting\n"
        f"{p2},steps,1200,count,2026-06-01T08:00:00+00:00,apple_watch_s9,active\n"
    )
    rows = list(rows_from_text(text))
    assert len(rows) == 2
    assert isinstance(rows[0]["value"], float)  # coerced from CSV string
    result = normalise_batch(rows, default_adapter=ADAPTER_NAME)
    assert len(result.accepted) == 2
    assert result.rejections == []


def test_csv_row_missing_unit_rejected() -> None:
    text = (
        "patient_id,metric_code,value,timestamp,source_device,context\n"
        f"{uuid4()},heart_rate,58,2026-06-01T07:30:00+00:00,apple_watch_s9,resting\n"
    )
    result = normalise_batch(rows_from_text(text), default_adapter=ADAPTER_NAME)
    assert len(result.rejections) == 1
    assert any(e["field"] == "unit" for e in result.rejections[0]["errors"])
