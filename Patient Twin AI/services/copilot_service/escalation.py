"""Clinician escalation queue (docs/07 §6, docs/10 T4.3, docs/15 T6.3).

The copilot enqueues every red-flag / high-severity `OutputContract` here via the
`EscalationSink` port. This queryable implementation lets a clinician list the open
queue and acknowledge an item. Two invariants (docs/15 T6.3 Don't):

  * Ack NEVER mutates or re-opens the underlying output — the acknowledgement is
    tracked in a wrapper (`EscalationRecord`), the immutable `OutputContract` is
    kept verbatim.
  * Every ack is written to the hash-chained audit trail as a CLINICIAN action,
    version-stamped from the output it acknowledges (docs/06 §7-8).

Dev wiring is in-memory; production swaps the store while keeping this interface.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

from core.audit import AuditWriter
from schemas.audit import AuditAction, AuditActor
from schemas.output_contract import OutputContract


class EscalationStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"


class EscalationRecord(BaseModel):
    """An escalated output plus its acknowledgement state (never the output itself)."""

    output: OutputContract
    status: EscalationStatus = EscalationStatus.OPEN
    acknowledged_by: str | None = None
    acknowledged_at: datetime | None = None


class EscalationNotFoundError(LookupError):
    """No escalation queued for the given output id."""


class EscalationQueue:
    """A queryable `EscalationSink`. `enqueue` is the copilot's write path; the rest
    is the clinician read/ack surface.
    """

    def __init__(self, audit_writer: AuditWriter) -> None:
        self._audit = audit_writer
        # Insertion-ordered (FIFO) so the clinician sees the oldest escalation first.
        self._records: dict[UUID, EscalationRecord] = {}

    # -- copilot write path --------------------------------------------------

    def enqueue(self, output: OutputContract) -> None:
        # Idempotent on output_id: an output is escalated once. The orchestrator only
        # calls this when `output.escalation.triggered`, so no extra guard is needed.
        if output.output_id in self._records:
            return
        self._records[output.output_id] = EscalationRecord(output=output)

    # -- clinician read/ack surface ------------------------------------------

    def list(
        self, *, status: EscalationStatus | None = EscalationStatus.OPEN
    ) -> list[EscalationRecord]:
        """Queued escalations, oldest first. `status=None` lists all."""
        records = list(self._records.values())
        if status is not None:
            records = [r for r in records if r.status is status]
        return records

    def get(self, output_id: UUID) -> EscalationRecord | None:
        return self._records.get(output_id)

    def acknowledge(self, output_id: UUID, *, clinician: str, now: datetime) -> EscalationRecord:
        """Record a clinician acknowledgement + audit it. Idempotent: acking an
        already-acknowledged item returns it unchanged and writes no second record.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        record = self._records.get(output_id)
        if record is None:
            raise EscalationNotFoundError(str(output_id))
        if record.status is EscalationStatus.ACKNOWLEDGED:
            return record

        acked = record.model_copy(
            update={
                "status": EscalationStatus.ACKNOWLEDGED,
                "acknowledged_by": clinician,
                "acknowledged_at": now,
            }
        )
        self._records[output_id] = acked
        self._audit.write(
            patient_id=record.output.patient_id,
            actor=AuditActor.CLINICIAN,
            action=AuditAction.ESCALATION_ACK,
            input_refs=[f"output:{output_id}"],
            output_refs=[f"ack:{clinician}"],
            versions=record.output.versions.model_dump(),
            timestamp=now,
        )
        return acked
