"""Consent lifecycle is audit-backed and deny-by-default (docs/04 §1, docs/07 §2)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from core.audit import AuditWriter, InMemoryAuditStore, verify_chain
from schemas.audit import AuditAction, AuditActor
from schemas.consent import ConsentScope
from services.governance_service.consent import ConsentLedger

NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


def _ledger() -> tuple[ConsentLedger, InMemoryAuditStore]:
    store = InMemoryAuditStore()
    return ConsentLedger(AuditWriter(store)), store


def test_unknown_patient_has_no_consent() -> None:
    ledger, _ = _ledger()
    assert ledger.get_consent(uuid4()) is None


def test_grant_records_consent_and_audits_it() -> None:
    ledger, store = _ledger()
    pid = uuid4()
    consent = ledger.grant(pid, scope=[ConsentScope.COPILOT], version="v1", now=NOW)

    assert consent.covers(ConsentScope.COPILOT)
    assert ledger.get_consent(pid) == consent
    assert len(store.records) == 1
    rec = store.records[0]
    assert rec.action == AuditAction.CONSENT_CHANGE
    assert rec.actor == AuditActor.PATIENT
    assert rec.patient_id == pid


def test_revoke_stops_processing_and_audits() -> None:
    ledger, store = _ledger()
    pid = uuid4()
    ledger.grant(pid, scope=[ConsentScope.COPILOT], version="v1", now=NOW)
    revoked = ledger.revoke(pid, now=NOW)

    assert revoked is not None
    assert revoked.revoked_at is not None
    assert not revoked.covers(ConsentScope.COPILOT)
    assert len(store.records) == 2  # grant + revoke
    verify_chain(store.records)


def test_revoke_is_idempotent_and_mints_no_spurious_record() -> None:
    ledger, store = _ledger()
    pid = uuid4()
    ledger.grant(pid, scope=[ConsentScope.VITALS], version="v1", now=NOW)
    ledger.revoke(pid, now=NOW)
    again = ledger.revoke(pid, now=NOW)

    assert again is not None and again.revoked_at is not None
    assert len(store.records) == 2  # grant + one revoke only


def test_revoke_unknown_patient_is_noop() -> None:
    ledger, store = _ledger()
    assert ledger.revoke(uuid4(), now=NOW) is None
    assert len(store.records) == 0


def test_grant_replaces_prior_consent() -> None:
    ledger, _ = _ledger()
    pid = uuid4()
    ledger.grant(pid, scope=[ConsentScope.VITALS], version="v1", now=NOW)
    updated = ledger.grant(
        pid, scope=[ConsentScope.VITALS, ConsentScope.COPILOT], version="v2", now=NOW
    )

    assert updated.version == "v2"
    assert updated.covers(ConsentScope.COPILOT)
