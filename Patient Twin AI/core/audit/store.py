"""In-memory audit store. Dependency-free (no SQLAlchemy), so the chain logic is
testable anywhere. The Postgres-backed store lives in `core.audit.sql_store` and
is imported only when a DB is actually wired up.
"""

from __future__ import annotations

from schemas.audit import AuditRecord

from .hashing import GENESIS_PREV_HASH


class InMemoryAuditStore:
    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    def head_hash(self) -> str:
        return self._records[-1].hash if self._records else GENESIS_PREV_HASH

    def append(self, record: AuditRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> list[AuditRecord]:
        return list(self._records)
