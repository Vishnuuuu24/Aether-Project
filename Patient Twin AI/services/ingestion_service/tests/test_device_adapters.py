"""HealthKit / Health Connect / Fitbit adapters map behind the shared normaliser."""

from __future__ import annotations

from uuid import uuid4

from services.ingestion_service.adapters import fitbit, health_connect, healthkit
from services.ingestion_service.normaliser import normalise_batch

TS = "2026-06-01T07:30:00+00:00"


def test_healthkit_maps_known_types_and_skips_unknown() -> None:
    pid = str(uuid4())
    samples = [
        {
            "type": "HKQuantityTypeIdentifierHeartRate",
            "value": 60,
            "startDate": TS,
            "sourceName": "apple_watch",
        },
        {"type": "HKQuantityTypeIdentifierStepCount", "value": 1000, "startDate": TS},
        {"type": "HKQuantityTypeIdentifierNotAThing", "value": 1, "startDate": TS},
    ]
    raws = list(healthkit.to_canonical(samples, patient_id=pid))
    assert len(raws) == 2  # unknown type skipped, not fabricated
    result = normalise_batch(raws, default_adapter=healthkit.ADAPTER_NAME)
    assert len(result.accepted) == 2


def test_fitbit_maps_and_normalises() -> None:
    pid = str(uuid4())
    points = [{"resource": "heart_rate", "value": 61, "dateTime": TS, "device": "charge6"}]
    result = normalise_batch(
        fitbit.to_canonical(points, patient_id=pid), default_adapter=fitbit.ADAPTER_NAME
    )
    assert len(result.accepted) == 1


def test_health_connect_maps_and_normalises() -> None:
    pid = str(uuid4())
    records = [{"recordType": "StepsRecord", "value": 500, "time": TS}]
    result = normalise_batch(
        health_connect.to_canonical(records, patient_id=pid),
        default_adapter=health_connect.ADAPTER_NAME,
    )
    assert len(result.accepted) == 1
