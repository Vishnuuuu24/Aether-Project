"""POST /v1/ingest/readings — batch, per-item 207 multi-status (docs/07 §3)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from core.audit import AuditWriter, InMemoryAuditStore
from schemas.consent import Consent, ConsentScope
from services.ingestion_service.app.main import app, get_service
from services.ingestion_service.consent import StaticConsentProvider
from services.ingestion_service.service import IngestionService
from services.ingestion_service.sink import InMemoryReadingSink

TS_ISO = "2026-06-01T07:30:00+00:00"


def reading_json(patient_id: UUID, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "patient_id": str(patient_id),
        "metric_code": "heart_rate",
        "value": 58,
        "unit": "bpm",
        "timestamp": TS_ISO,
        "source_device": "apple_watch_s9",
        "context": "resting",
    }
    data.update(overrides)
    return data


@pytest.fixture
def client_and_patient() -> Iterator[tuple[TestClient, UUID]]:
    pid = uuid4()
    provider = StaticConsentProvider()
    provider.grant(
        pid,
        Consent(
            scope=[ConsentScope.VITALS], version="v1", granted_at=datetime(2026, 1, 1, tzinfo=UTC)
        ),
    )
    service = IngestionService(
        consent_provider=provider,
        sink=InMemoryReadingSink(),
        audit_writer=AuditWriter(InMemoryAuditStore()),
    )
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app), pid
    app.dependency_overrides.clear()


def test_all_valid_returns_200(client_and_patient: tuple[TestClient, UUID]) -> None:
    client, pid = client_and_patient
    resp = client.post("/v1/ingest/readings", json=[reading_json(pid), reading_json(pid)])
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["accepted"]) == 2
    assert body["rejected"] == []


def test_mixed_batch_returns_207_with_field_error(
    client_and_patient: tuple[TestClient, UUID],
) -> None:
    client, pid = client_and_patient
    good = reading_json(pid)
    bad = reading_json(pid, timestamp="2026-06-01T07:30:00")  # naive → rejected
    resp = client.post("/v1/ingest/readings", json=[good, bad])
    assert resp.status_code == 207
    body = resp.json()
    assert len(body["accepted"]) == 1
    assert len(body["rejected"]) == 1
    assert body["rejected"][0]["index"] == 1
    assert any(e["field"] == "timestamp" for e in body["rejected"][0]["errors"])


def test_unconsented_patient_rejected(client_and_patient: tuple[TestClient, UUID]) -> None:
    client, _ = client_and_patient
    resp = client.post("/v1/ingest/readings", json=[reading_json(uuid4())])
    assert resp.status_code == 207
    assert resp.json()["rejected"][0]["errors"][0]["field"] == "consent"


def test_healthz(client_and_patient: tuple[TestClient, UUID]) -> None:
    client, _ = client_and_patient
    assert client.get("/healthz").json() == {"status": "alive"}
