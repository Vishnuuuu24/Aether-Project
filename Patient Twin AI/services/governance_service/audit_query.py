"""Read side of the hash-chained audit trail. docs/07 §7.

`core.audit` owns writing and chain verification; this module is the query layer
behind `GET /v1/audit`. It filters the append-only records and, crucially,
re-verifies the returned run so a caller can *trust* what it reads — an audit that
can't prove it wasn't tampered with is worthless.

"Audit reconstructs any output" (T5.1 DoD): `records_for_output` returns every
record referencing a given `output_id`, ordered, which is the reconstruction path.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from core.audit import verify_chain
from schemas.audit import AuditAction, AuditRecord


class AuditReader(Protocol):
    """Read port over the audit chain. `InMemoryAuditStore` satisfies it; the
    Postgres-backed store exposes the same ordered `records`."""

    @property
    def records(self) -> list[AuditRecord]: ...


def query_audit(
    reader: AuditReader,
    *,
    patient_id: UUID | None = None,
    action: AuditAction | None = None,
    since: datetime | None = None,
    output_id: UUID | None = None,
) -> list[AuditRecord]:
    """Filtered slice of the audit trail, in chain order.

    The full chain is verified first (raises `AuditChainError` on tampering)
    BEFORE any filter is applied — you cannot get a "clean" filtered view over a
    corrupted chain.
    """
    verify_chain(reader.records)
    out_ref = str(output_id) if output_id is not None else None
    result: list[AuditRecord] = []
    for rec in reader.records:
        if patient_id is not None and rec.patient_id != patient_id:
            continue
        if action is not None and rec.action != action:
            continue
        if since is not None and rec.timestamp < since:
            continue
        if out_ref is not None and out_ref not in rec.output_refs and out_ref not in rec.input_refs:
            continue
        result.append(rec)
    return result


def records_for_output(reader: AuditReader, output_id: UUID) -> list[AuditRecord]:
    """Every audit record that references `output_id` — the reconstruction of a
    single output's provenance (T5.1 DoD)."""
    return query_audit(reader, output_id=output_id)
