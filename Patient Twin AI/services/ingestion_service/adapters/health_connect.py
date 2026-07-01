"""Android Health Connect adapter (docs/07 §3).

Maps Health Connect records to canonical readings behind the shared normaliser.
Starting map to be completed against real Health Connect payloads.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

ADAPTER_NAME = "health_connect"

# Health Connect record type → (canonical metric_code, canonical unit).
_TYPE_MAP: dict[str, tuple[str, str]] = {
    "HeartRateRecord": ("heart_rate", "bpm"),
    "StepsRecord": ("steps", "count"),
    "OxygenSaturationRecord": ("spo2", "%"),
    "RespiratoryRateRecord": ("respiratory_rate", "breaths/min"),
    "SkinTemperatureRecord": ("skin_temp", "celsius"),
}


def to_canonical(records: Iterable[dict[str, Any]], *, patient_id: str) -> Iterator[dict[str, Any]]:
    for record in records:
        mapped = _TYPE_MAP.get(record.get("recordType", ""))
        if mapped is None:
            continue
        metric_code, unit = mapped
        yield {
            "patient_id": patient_id,
            "metric_code": metric_code,
            "value": record.get("value"),
            "unit": unit,
            "timestamp": record.get("time"),
            "source_device": record.get("dataOrigin", "health_connect"),
            "context": record.get("context", "unknown"),
            "ingest_adapter": ADAPTER_NAME,
        }
