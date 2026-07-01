"""Ingestion orchestration: consent gate (deny-by-default), sink handoff, audit."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from core.audit import AuditWriter, InMemoryAuditStore, verify_chain
from schemas.consent import Consent, ConsentScope
from services.ingestion_service.consent import StaticConsentProvider
from services.ingestion_service.service import IngestionService
from services.ingestion_service.sink import InMemoryReadingSink

TS = datetime(2026, 6, 1, 7, 30, tzinfo=UTC)


def raw(patient_id: UUID, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "patient_id": str(patient_id),
        "metric_code": "heart_rate",
        "value": 58,
        "unit": "bpm",
        "timestamp": TS,
        "source_device": "apple_watch_s9",
        "context": "resting",
    }
    data.update(overrides)
    return data


def make_service(
    *granted: UUID,
) -> tuple[IngestionService, InMemoryReadingSink, InMemoryAuditStore]:
    provider = StaticConsentProvider()
    for pid in granted:
        provider.grant(pid, Consent(scope=[ConsentScope.VITALS], version="v1", granted_at=TS))
    sink = InMemoryReadingSink()
    store = InMemoryAuditStore()
    service = IngestionService(
        consent_provider=provider, sink=sink, audit_writer=AuditWriter(store)
    )
    return service, sink, store


def test_consented_readings_emitted_and_audited() -> None:
    pid = uuid4()
    service, sink, store = make_service(pid)
    result = service.ingest([raw(pid)], adapter="csv")
    assert len(result.accepted) == 1
    assert len(sink.readings) == 1
    assert len(store.records) == 1
    verify_chain(store.records)
    assert store.records[0].action.value == "ingest"


def test_unconsented_patient_rejected_deny_by_default() -> None:
    pid = uuid4()
    service, sink, store = make_service()  # nobody granted
    result = service.ingest([raw(pid)], adapter="csv")
    assert result.accepted == []
    assert result.rejected[0]["errors"][0]["field"] == "consent"
    assert sink.readings == []
    assert store.records == []  # nothing accepted → no audit


def test_mixed_patients_partition_by_consent() -> None:
    ok, denied = uuid4(), uuid4()
    service, sink, store = make_service(ok)
    result = service.ingest([raw(ok), raw(denied)], adapter="csv")
    assert len(result.accepted) == 1
    assert len(result.rejected) == 1
    assert len(sink.readings) == 1


def test_malformed_reading_rejected_before_consent() -> None:
    pid = uuid4()
    service, _, _ = make_service(pid)
    bad = raw(pid)
    del bad["unit"]
    result = service.ingest([bad], adapter="csv")
    fields = {e["field"] for e in result.rejected[0]["errors"]}
    assert "unit" in fields
    assert "consent" not in fields
