"""Hash chain — the T0.2 DoD test: 'audit chain verifies' (and detects tampering).

Uses the dependency-free in-memory store so the chain logic is exercised without a
database. The Postgres-backed store is verified end-to-end in core/db/tests.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from core.audit import (
    GENESIS_PREV_HASH,
    AuditChainError,
    AuditWriter,
    InMemoryAuditStore,
    record_hash,
    verify_chain,
)
from schemas.audit import AuditAction, AuditActor


def build_chain(n: int = 3):
    store = InMemoryAuditStore()
    writer = AuditWriter(store)
    pid = uuid4()
    records = [
        writer.write(
            patient_id=pid,
            actor=AuditActor.SYSTEM,
            action=AuditAction.INGEST,
            input_refs=[f"reading:{i}"],
            versions={"model": "m1"},
        )
        for i in range(n)
    ]
    return store, records


def test_head_starts_at_genesis() -> None:
    store = InMemoryAuditStore()
    assert store.head_hash() == GENESIS_PREV_HASH


def test_first_record_links_to_genesis() -> None:
    _, records = build_chain(1)
    assert records[0].prev_hash == GENESIS_PREV_HASH


def test_records_link_head_advances() -> None:
    store, records = build_chain(3)
    for prev, cur in zip(records, records[1:], strict=False):
        assert cur.prev_hash == prev.hash
    assert store.head_hash() == records[-1].hash


def test_each_record_hash_matches_content() -> None:
    _, records = build_chain(4)
    for rec in records:
        assert rec.hash == record_hash(rec)


def test_intact_chain_verifies() -> None:
    _, records = build_chain(5)
    verify_chain(records)  # must not raise


def test_tampered_content_detected_at_index() -> None:
    _, records = build_chain(3)
    records[1] = records[1].model_copy(update={"input_refs": ["HACKED"]})
    with pytest.raises(AuditChainError) as exc:
        verify_chain(records)
    assert exc.value.index == 1


def test_removed_record_breaks_link() -> None:
    _, records = build_chain(3)
    broken = [records[0], records[2]]  # drop the middle → prev_hash no longer lines up
    with pytest.raises(AuditChainError) as exc:
        verify_chain(broken)
    assert exc.value.index == 1


def test_reordered_records_detected() -> None:
    _, records = build_chain(3)
    with pytest.raises(AuditChainError):
        verify_chain([records[1], records[0], records[2]])
