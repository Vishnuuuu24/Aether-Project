"""Apple HealthKit adapter (docs/07 §3).

Maps HealthKit quantity samples to canonical readings, then through the shared
normaliser. The type/unit map below is a documented starting point — the full
`HKQuantityTypeIdentifier` set and exact units are completed against real
HealthKit exports. Because everything funnels through the normaliser, extending it
is a mapping change, not a new call site.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

ADAPTER_NAME = "healthkit"

# HKQuantityTypeIdentifier → (canonical metric_code, canonical unit).
_TYPE_MAP: dict[str, tuple[str, str]] = {
    "HKQuantityTypeIdentifierHeartRate": ("heart_rate", "bpm"),
    "HKQuantityTypeIdentifierStepCount": ("steps", "count"),
    "HKQuantityTypeIdentifierOxygenSaturation": ("spo2", "%"),
    "HKQuantityTypeIdentifierRespiratoryRate": ("respiratory_rate", "breaths/min"),
    "HKQuantityTypeIdentifierBodyTemperature": ("skin_temp", "celsius"),
}


def to_canonical(samples: Iterable[dict[str, Any]], *, patient_id: str) -> Iterator[dict[str, Any]]:
    """Map HealthKit samples ({type, value, startDate, sourceName, context?}).

    Unmapped sample types are skipped (not fabricated). Missing fields fall through
    to the normaliser, which rejects them with a field error.
    """
    for sample in samples:
        mapped = _TYPE_MAP.get(sample.get("type", ""))
        if mapped is None:
            continue
        metric_code, unit = mapped
        yield {
            "patient_id": patient_id,
            "metric_code": metric_code,
            "value": sample.get("value"),
            "unit": unit,
            "timestamp": sample.get("startDate"),
            "source_device": sample.get("sourceName", "apple_health"),
            "context": sample.get("context", "unknown"),
            "ingest_adapter": ADAPTER_NAME,
        }
