"""Audit adapter: emit a hash-chained record for every copilot output (docs/06 §8).

`AuditWriterSink` implements the `AuditSink` port by writing a `POLICY_DECISION`
record through `core.audit.AuditWriter` — so every user-facing output (approved,
downgraded, suppressed, abstained, escalated) is linked into the append-only chain
and any output can later be reconstructed/verified (docs/04 §7). The store behind the
writer is injected: in-memory for dev, the Postgres-backed store in production.
"""

from __future__ import annotations

from core.audit import AuditWriter
from schemas.audit import AuditAction, AuditActor
from schemas.output_contract import OutputContract


class AuditWriterSink:
    def __init__(self, writer: AuditWriter) -> None:
        self._writer = writer

    def record(self, output: OutputContract) -> None:
        self._writer.write(
            patient_id=output.patient_id,
            actor=AuditActor.SYSTEM,
            action=AuditAction.POLICY_DECISION,
            input_refs=[e.ref for e in output.evidence],
            output_refs=[str(output.output_id)],
            versions=output.versions.model_dump(),
            timestamp=output.created_at,
        )
