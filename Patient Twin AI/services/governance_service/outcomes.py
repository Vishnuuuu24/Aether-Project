"""Outer-loop outcome capture. docs/11 §3, docs/07 §7.

Records real clinical outcomes linked to the prior outputs and versions that
preceded them. v1 stores only — this is the labelled signal for LATER human-gated
retraining, never a live feedback loop (CLAUDE.md principle 5).

Each capture mints an `OUTCOME_CAPTURE` audit record, so the outcome is itself
part of the tamper-evident chain and joinable back to the outputs it references.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from core.audit import AuditWriter
from schemas.audit import AuditAction, AuditActor
from schemas.outcome import Outcome, OutcomeSource

_ACTOR_BY_SOURCE = {
    OutcomeSource.CLINICIAN: AuditActor.CLINICIAN,
    OutcomeSource.EHR_IMPORT: AuditActor.SYSTEM,
    OutcomeSource.PATIENT_REPORTED: AuditActor.PATIENT,
}


class OutcomeStore:
    def __init__(self, audit_writer: AuditWriter) -> None:
        self._audit_writer = audit_writer
        self._by_id: dict[UUID, Outcome] = {}

    def record(self, outcome: Outcome, *, now: datetime | None = None) -> Outcome:
        """Persist an outcome and audit the capture, linking it to prior outputs."""
        self._by_id[outcome.outcome_id] = outcome
        self._audit_writer.write(
            patient_id=outcome.patient_id,
            actor=_ACTOR_BY_SOURCE[outcome.source],
            action=AuditAction.OUTCOME_CAPTURE,
            input_refs=[str(oid) for oid in outcome.linked_output_ids],
            output_refs=[f"outcome:{outcome.outcome_id}"],
            versions=outcome.versions,
            timestamp=now or outcome.recorded_at or datetime.now(UTC),
        )
        return outcome

    def get(self, outcome_id: UUID) -> Outcome | None:
        return self._by_id.get(outcome_id)

    def for_patient(self, patient_id: UUID) -> list[Outcome]:
        return [o for o in self._by_id.values() if o.patient_id == patient_id]
