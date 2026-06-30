"""Hash-chained, append-only audit writer + verifier (docs/04 §7, docs/06 §8).

The writer is the ONLY way to mint an `AuditRecord`: it reads the chain head,
computes the linked hash, and appends — so callers cannot forge `prev_hash`/`hash`.
A `Store` abstracts where the chain lives (in-memory for tests, Postgres in prod),
which keeps the hashing logic identical everywhere and lets the chain be made
per-patient later without touching this code.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from schemas.audit import AuditAction, AuditActor, AuditRecord

from .errors import AuditChainError
from .hashing import GENESIS_PREV_HASH, compute_hash


def _payload(record: AuditRecord) -> dict[str, Any]:
    """The hash preimage dict: every field except `hash`."""
    return {
        "audit_id": record.audit_id,
        "patient_id": record.patient_id,
        "actor": record.actor,
        "action": record.action,
        "input_refs": record.input_refs,
        "output_refs": record.output_refs,
        "versions": record.versions,
        "timestamp": record.timestamp,
        "prev_hash": record.prev_hash,
    }


def record_hash(record: AuditRecord) -> str:
    """Recompute the hash a record *should* have. Used by the writer and verifier."""
    return compute_hash(_payload(record))


class Store(Protocol):
    def head_hash(self) -> str: ...
    def append(self, record: AuditRecord) -> None: ...


class AuditWriter:
    """Mints linked, append-only audit records against a `Store`."""

    def __init__(self, store: Store) -> None:
        self._store = store

    def write(
        self,
        *,
        patient_id: UUID,
        actor: AuditActor,
        action: AuditAction,
        input_refs: Iterable[str] = (),
        output_refs: Iterable[str] = (),
        versions: dict[str, str] | None = None,
        timestamp: datetime | None = None,
    ) -> AuditRecord:
        prev_hash = self._store.head_hash()
        ts = timestamp or datetime.now(UTC)
        if ts.tzinfo is None:
            raise ValueError("audit timestamp must be timezone-aware")
        fields: dict[str, Any] = {
            "audit_id": uuid4(),
            "patient_id": patient_id,
            "actor": actor,
            "action": action,
            "input_refs": list(input_refs),
            "output_refs": list(output_refs),
            "versions": dict(versions or {}),
            "timestamp": ts,
            "prev_hash": prev_hash,
        }
        digest = compute_hash(fields)
        record = AuditRecord(**fields, hash=digest)
        self._store.append(record)
        return record


def verify_record(record: AuditRecord, expected_prev_hash: str) -> None:
    """Raise AuditChainError if `record` does not link to `expected_prev_hash`
    or its own hash does not match its content."""
    if record.prev_hash != expected_prev_hash:
        raise AuditChainError(
            f"prev_hash mismatch: expected {expected_prev_hash}, got {record.prev_hash}"
        )
    recomputed = record_hash(record)
    if record.hash != recomputed:
        raise AuditChainError(f"hash mismatch: stored {record.hash}, recomputed {recomputed}")


def verify_chain(records: Sequence[AuditRecord]) -> None:
    """Verify an ordered run of records links from genesis. Raises AuditChainError
    (with `.index` at the first break) on any tampering; returns None if intact.
    """
    expected_prev = GENESIS_PREV_HASH
    for i, record in enumerate(records):
        try:
            verify_record(record, expected_prev)
        except AuditChainError as exc:
            raise AuditChainError(str(exc), index=i) from exc
        expected_prev = record.hash
