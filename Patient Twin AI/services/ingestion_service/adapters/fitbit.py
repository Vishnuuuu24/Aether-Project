"""Fitbit Web API adapter (docs/07 §3).

Maps Fitbit intraday/summary series to canonical readings behind the shared
normaliser. Starting map to be completed against real Fitbit API responses.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

ADAPTER_NAME = "fitbit"

# Fitbit resource key → (canonical metric_code, canonical unit).
_RESOURCE_MAP: dict[str, tuple[str, str]] = {
    "heart_rate": ("heart_rate", "bpm"),
    "steps": ("steps", "count"),
    "spo2": ("spo2", "%"),
    "br": ("respiratory_rate", "breaths/min"),
    "skin_temp": ("skin_temp", "celsius"),
}


def to_canonical(points: Iterable[dict[str, Any]], *, patient_id: str) -> Iterator[dict[str, Any]]:
    """Map Fitbit datapoints ({resource, value, dateTime, device?, context?})."""
    for point in points:
        mapped = _RESOURCE_MAP.get(point.get("resource", ""))
        if mapped is None:
            continue
        metric_code, unit = mapped
        yield {
            "patient_id": patient_id,
            "metric_code": metric_code,
            "value": point.get("value"),
            "unit": unit,
            "timestamp": point.get("dateTime"),
            "source_device": point.get("device", "fitbit"),
            "context": point.get("context", "unknown"),
            "ingest_adapter": ADAPTER_NAME,
        }
