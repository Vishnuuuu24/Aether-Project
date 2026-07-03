"""Outcome capture stores the label and audits it against prior outputs (docs/11 §3)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from core.audit import AuditWriter, InMemoryAuditStore
from schemas.audit import AuditAction, AuditActor
from schemas.outcome import Outcome, OutcomeSource, OutcomeType
from services.governance_service.outcomes import OutcomeStore

NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


def _outcome(**overrides: object) -> Outcome:
    base: dict[str, object] = {
        "patient_id": uuid4(),
        "outcome_type": OutcomeType.ADMISSION,
        "occurred_at": NOW,
        "detail": "admitted for chest pain",
        "linked_output_ids": [uuid4()],
        "versions": {"model": "m1", "ruleset": "r1"},
        "source": OutcomeSource.CLINICIAN,
        "recorded_at": NOW,
    }
    base.update(overrides)
    return Outcome(**base)  # type: ignore[arg-type]


def test_record_persists_and_is_retrievable() -> None:
    store = OutcomeStore(AuditWriter(InMemoryAuditStore()))
    outcome = _outcome()
    store.record(outcome)
    assert store.get(outcome.outcome_id) == outcome
    assert store.for_patient(outcome.patient_id) == [outcome]


def test_record_audits_capture_linked_to_prior_outputs() -> None:
    audit_store = InMemoryAuditStore()
    store = OutcomeStore(AuditWriter(audit_store))
    output_id = uuid4()
    outcome = _outcome(linked_output_ids=[output_id])
    store.record(outcome)

    assert len(audit_store.records) == 1
    rec = audit_store.records[0]
    assert rec.action == AuditAction.OUTCOME_CAPTURE
    assert rec.actor == AuditActor.CLINICIAN  # clinician-sourced
    assert str(output_id) in rec.input_refs
    assert f"outcome:{outcome.outcome_id}" in rec.output_refs


def test_source_maps_to_audit_actor() -> None:
    audit_store = InMemoryAuditStore()
    store = OutcomeStore(AuditWriter(audit_store))
    store.record(_outcome(source=OutcomeSource.EHR_IMPORT))
    assert audit_store.records[0].actor == AuditActor.SYSTEM


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _outcome(occurred_at=datetime(2026, 7, 3, 12, 0))  # noqa: DTZ001 — intentional
