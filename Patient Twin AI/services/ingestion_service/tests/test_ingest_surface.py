"""Ingestion HTTP surface beyond /readings: adapter webhooks + dataset replay
(docs/07 §3; docs/15 T6.4). Everything funnels through the one normaliser + the
deny-by-default consent gate.
"""

from __future__ import annotations

import pickle
from datetime import UTC, datetime
from pathlib import Path
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

TS = "2026-06-01T07:30:00+00:00"


def _service_with_consent(pid: UUID | None) -> IngestionService:
    consent = StaticConsentProvider()
    if pid is not None:
        consent.grant(
            pid,
            Consent(
                scope=[ConsentScope.VITALS],
                version="v1",
                granted_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )
    return IngestionService(
        consent_provider=consent,
        sink=InMemoryReadingSink(),
        audit_writer=AuditWriter(InMemoryAuditStore()),
    )


def _client(service: IngestionService) -> TestClient:
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def _dalia_sample() -> dict[str, Any]:
    # PPG-DaLiA shape (lists, not numpy): 4 HR labels + 3 wrist-TEMP samples.
    return {
        "label": [70.0, 72.0, 95.0, 110.0],
        "activity": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4],
        "signal": {"wrist": {"TEMP": [33.0, 33.1, 33.2]}},
    }


# -- webhook --------------------------------------------------------------------


def test_webhook_maps_and_ingests_with_consent() -> None:
    pid = uuid4()
    payload = {
        "patient_id": str(pid),
        "items": [
            {
                "type": "HKQuantityTypeIdentifierHeartRate",
                "value": 60,
                "startDate": TS,
                "sourceName": "apple_watch",
            }
        ],
    }
    try:
        resp = _client(_service_with_consent(pid)).post(
            "/v1/ingest/adapters/healthkit/webhook", json=payload
        )
        assert resp.status_code == 200
        assert len(resp.json()["accepted"]) == 1
    finally:
        app.dependency_overrides.clear()


def test_webhook_unknown_adapter_is_404() -> None:
    try:
        resp = _client(_service_with_consent(uuid4())).post(
            "/v1/ingest/adapters/nope/webhook",
            json={"patient_id": str(uuid4()), "items": []},
        )
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_webhook_without_consent_rejects_deny_by_default() -> None:
    pid = uuid4()
    payload = {
        "patient_id": str(pid),
        "items": [{"resource": "heart_rate", "value": 61, "dateTime": TS, "device": "charge6"}],
    }
    try:
        resp = _client(_service_with_consent(None)).post(  # no consent seeded
            "/v1/ingest/adapters/fitbit/webhook", json=payload
        )
        assert resp.status_code == 207  # rejected, not silently processed
        assert resp.json()["accepted"] == []
        assert resp.json()["rejected"]
    finally:
        app.dependency_overrides.clear()


# -- replay ---------------------------------------------------------------------


def test_replay_streams_when_dataset_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = uuid4()
    ds = tmp_path / "PPG-DaLiA"
    ds.mkdir()
    with open(ds / "S1.pkl", "wb") as handle:
        pickle.dump(_dalia_sample(), handle)
    monkeypatch.setenv("INGEST_DATASETS_DIR", str(tmp_path))

    try:
        resp = _client(_service_with_consent(pid)).post(
            "/v1/ingest/replay",
            json={"dataset": "PPG-DaLiA", "patient_id": str(pid), "speed": 1.0},
        )
        assert resp.status_code == 200
        assert len(resp.json()["accepted"]) == 7  # 4 heart_rate + 3 skin_temp
    finally:
        app.dependency_overrides.clear()


def test_replay_missing_dataset_file_is_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("INGEST_DATASETS_DIR", str(tmp_path))
    try:
        resp = _client(_service_with_consent(uuid4())).post(
            "/v1/ingest/replay",
            json={"dataset": "PPG-DaLiA", "patient_id": str(uuid4()), "speed": 1.0},
        )
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_replay_unknown_dataset_is_422(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A file exists but the dataset name is unsupported → the adapter raises → 422.
    bogus = tmp_path / "BOGUS"
    bogus.mkdir()
    (bogus / "S1.pkl").write_bytes(pickle.dumps({}))
    monkeypatch.setenv("INGEST_DATASETS_DIR", str(tmp_path))
    try:
        resp = _client(_service_with_consent(uuid4())).post(
            "/v1/ingest/replay",
            json={"dataset": "BOGUS", "patient_id": str(uuid4()), "speed": 1.0},
        )
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_replay_disabled_is_403(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INGEST_ENABLE_REPLAY", "0")
    try:
        resp = _client(_service_with_consent(uuid4())).post(
            "/v1/ingest/replay",
            json={"dataset": "PPG-DaLiA", "patient_id": str(uuid4()), "speed": 1.0},
        )
        assert resp.status_code == 403
    finally:
        app.dependency_overrides.clear()
