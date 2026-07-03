"""Audit query filters, re-verifies the chain, and reconstructs an output (T5.1 DoD)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from core.audit import AuditWriter, InMemoryAuditStore
from core.audit.errors import AuditChainError
from schemas.audit import AuditAction, AuditActor
from services.governance_service.audit_query import query_audit, records_for_output

T0 = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


def _seed() -> tuple[InMemoryAuditStore, dict[str, object]]:
    store = InMemoryAuditStore()
    writer = AuditWriter(store)
    pid_a, pid_b = uuid4(), uuid4()
    output_id = uuid4()
    writer.write(
        patient_id=pid_a,
        actor=AuditActor.SYSTEM,
        action=AuditAction.POLICY_DECISION,
        output_refs=[str(output_id)],
        timestamp=T0,
    )
    writer.write(
        patient_id=pid_a,
        actor=AuditActor.PATIENT,
        action=AuditAction.CONSENT_CHANGE,
        timestamp=T0 + timedelta(hours=1),
    )
    writer.write(
        patient_id=pid_b,
        actor=AuditActor.SYSTEM,
        action=AuditAction.POLICY_DECISION,
        timestamp=T0 + timedelta(hours=2),
    )
    return store, {"pid_a": pid_a, "pid_b": pid_b, "output_id": output_id}


def test_filter_by_patient() -> None:
    store, ids = _seed()
    recs = query_audit(store, patient_id=ids["pid_a"])  # type: ignore[arg-type]
    assert len(recs) == 2
    assert all(r.patient_id == ids["pid_a"] for r in recs)


def test_filter_by_action() -> None:
    store, _ = _seed()
    recs = query_audit(store, action=AuditAction.POLICY_DECISION)
    assert len(recs) == 2
    assert all(r.action == AuditAction.POLICY_DECISION for r in recs)


def test_filter_by_since_is_inclusive_of_later_records() -> None:
    store, _ = _seed()
    recs = query_audit(store, since=T0 + timedelta(hours=1))
    assert len(recs) == 2  # drops the T0 record only


def test_reconstruct_output_finds_its_records() -> None:
    store, ids = _seed()
    recs = records_for_output(store, ids["output_id"])  # type: ignore[arg-type]
    assert len(recs) == 1
    assert str(ids["output_id"]) in recs[0].output_refs


def test_query_raises_on_tampered_chain() -> None:
    store, _ = _seed()
    # Tamper: mutate a stored record's content so its hash no longer matches.
    store._records[0] = store._records[0].model_copy(update={"actor": AuditActor.CLINICIAN})
    with pytest.raises(AuditChainError):
        query_audit(store)
