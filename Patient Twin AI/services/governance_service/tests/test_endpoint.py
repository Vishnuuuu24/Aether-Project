"""governance-service HTTP surface (docs/07 §2 & §7).

Uses dependency overrides so each test runs against a fresh in-memory wiring that
shares ONE audit store across consent / outcomes / audit — which is what lets the
DoD test record an outcome and then reconstruct it from the audit trail.
"""

from __future__ import annotations

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from core.audit import AuditWriter, InMemoryAuditStore
from schemas.audit import AuditAction
from services.governance_service.app.main import (
    app,
    get_audit_store,
    get_consent_ledger,
    get_outcome_store,
)
from services.governance_service.consent import ConsentLedger
from services.governance_service.outcomes import OutcomeStore


@pytest.fixture
def client() -> Iterator[TestClient]:
    audit_store = InMemoryAuditStore()
    writer = AuditWriter(audit_store)
    ledger = ConsentLedger(writer)
    outcomes = OutcomeStore(writer)
    app.dependency_overrides[get_consent_ledger] = lambda: ledger
    app.dependency_overrides[get_outcome_store] = lambda: outcomes
    app.dependency_overrides[get_audit_store] = lambda: audit_store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_consent_grant_get_revoke_roundtrip(client: TestClient) -> None:
    pid = str(uuid4())
    r = client.post(f"/v1/patients/{pid}/consent", json={"scope": ["copilot"], "version": "v1"})
    assert r.status_code == 200
    assert r.json()["scope"] == ["copilot"]

    r = client.get(f"/v1/patients/{pid}/consent")
    assert r.status_code == 200
    assert r.json()["revoked_at"] is None

    r = client.delete(f"/v1/patients/{pid}/consent")
    assert r.status_code == 200
    assert r.json()["revoked_at"] is not None


def test_get_consent_unknown_patient_is_404(client: TestClient) -> None:
    assert client.get(f"/v1/patients/{uuid4()}/consent").status_code == 404


def test_revoke_unknown_patient_is_404(client: TestClient) -> None:
    assert client.delete(f"/v1/patients/{uuid4()}/consent").status_code == 404


def test_versions_endpoint_returns_stamp(client: TestClient) -> None:
    r = client.get("/v1/versions")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"model", "ruleset", "prompt", "baseline_engine", "schema"}


def test_outcome_capture_then_reconstruct_from_audit(client: TestClient) -> None:
    """T5.1 DoD: an outcome is recorded against a prior output, and the audit trail
    reconstructs that output's provenance."""
    pid = str(uuid4())
    output_id = str(uuid4())
    r = client.post(
        "/v1/outcomes",
        json={
            "patient_id": pid,
            "outcome_type": "admission",
            "occurred_at": "2026-07-03T12:00:00Z",
            "detail": "admitted for chest pain",
            "linked_output_ids": [output_id],
            "versions": {"model": "m1"},
            "source": "clinician",
        },
    )
    assert r.status_code == 201
    outcome = r.json()
    assert outcome["linked_output_ids"] == [output_id]

    # Reconstruct: the audit trail, filtered to that output, shows the capture.
    r = client.get("/v1/audit", params={"output_id": output_id})
    assert r.status_code == 200
    recs = r.json()
    assert len(recs) == 1
    assert recs[0]["action"] == AuditAction.OUTCOME_CAPTURE.value
    assert output_id in recs[0]["input_refs"]


def test_audit_filter_by_patient(client: TestClient) -> None:
    pid = str(uuid4())
    client.post(f"/v1/patients/{pid}/consent", json={"scope": ["vitals"], "version": "v1"})
    r = client.get("/v1/audit", params={"patient_id": pid})
    assert r.status_code == 200
    recs = r.json()
    assert len(recs) == 1
    assert recs[0]["action"] == AuditAction.CONSENT_CHANGE.value
