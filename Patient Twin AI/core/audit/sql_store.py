"""Postgres-backed audit store for the single global chain.

Concurrency: a global chain requires that "read head → append" be serialised, or
two writers could compute the same `prev_hash` and fork the chain. We take a
transaction-scoped Postgres advisory lock in `head_hash()`; because `head_hash()`
and `append()` run in the same caller transaction, the lock is held across both
and released on commit/rollback.

The store does NOT commit — the caller owns the transaction boundary (so the audit
write can be atomic with the mutation it records).
"""

from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from schemas.audit import AuditRecord

from .hashing import GENESIS_PREV_HASH

# Arbitrary fixed key identifying the global audit chain lock namespace.
_AUDIT_CHAIN_LOCK_KEY = 0x4155_4449_5400  # "AUDIT\0"


class SqlAlchemyAuditStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def head_hash(self) -> str:
        from core.db.models import AuditLog

        # Serialise writers on the global chain for the rest of this transaction.
        self._session.execute(
            text("SELECT pg_advisory_xact_lock(:k)"), {"k": _AUDIT_CHAIN_LOCK_KEY}
        )
        last = self._session.execute(
            select(AuditLog.hash).order_by(AuditLog.seq.desc()).limit(1)
        ).scalar_one_or_none()
        return last or GENESIS_PREV_HASH

    def append(self, record: AuditRecord) -> None:
        from core.db.models import AuditLog

        self._session.add(
            AuditLog(
                audit_id=record.audit_id,
                patient_id=record.patient_id,
                actor=record.actor.value,
                action=record.action.value,
                input_refs=record.input_refs,
                output_refs=record.output_refs,
                versions=record.versions,
                timestamp=record.timestamp,
                prev_hash=record.prev_hash,
                hash=record.hash,
            )
        )
        self._session.flush()  # assign seq / surface integrity errors now
