"""Canonical hashing is deterministic and order-independent (docs/04 §7)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from core.audit import GENESIS_PREV_HASH, compute_hash

PID = UUID("00000000-0000-0000-0000-000000000001")
AID = UUID("00000000-0000-0000-0000-0000000000aa")
TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def payload(**overrides: object) -> dict:
    base = {
        "audit_id": AID,
        "patient_id": PID,
        "actor": "system",
        "action": "ingest",
        "input_refs": ["reading:1"],
        "output_refs": [],
        "versions": {"model": "m1", "ruleset": "r1"},
        "timestamp": TS,
        "prev_hash": GENESIS_PREV_HASH,
    }
    base.update(overrides)
    return base


def test_genesis_is_64_hex_zeros() -> None:
    assert GENESIS_PREV_HASH == "0" * 64


def test_deterministic() -> None:
    assert compute_hash(payload()) == compute_hash(payload())
    assert len(compute_hash(payload())) == 64


def test_key_insertion_order_irrelevant() -> None:
    a = {"actor": "system", "action": "ingest", "prev_hash": GENESIS_PREV_HASH}
    b = {"prev_hash": GENESIS_PREV_HASH, "action": "ingest", "actor": "system"}
    assert compute_hash(a) == compute_hash(b)


def test_any_field_change_changes_hash() -> None:
    h = compute_hash(payload())
    assert compute_hash(payload(input_refs=["reading:2"])) != h
    assert compute_hash(payload(action="sqi")) != h
    assert compute_hash(payload(prev_hash="f" * 64)) != h


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValueError):
        compute_hash(payload(timestamp=datetime(2026, 1, 1, 12, 0)))  # no tzinfo
