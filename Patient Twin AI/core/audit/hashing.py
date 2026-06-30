"""Deterministic hashing for the audit chain (docs/04 §7).

Decision (see core/README.md): SHA-256 over the full audit record MINUS the
`hash` field, serialised as canonical compact JSON — keys sorted, UTC RFC3339
timestamps, UUIDs/enums stringified. The preimage includes `prev_hash`, which is
what binds each record to its predecessor. Genesis `prev_hash` is 64 zeros.

The canonical form must never drift: any change here re-keys the whole chain.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

GENESIS_PREV_HASH = "0" * 64


def _canonical_default(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("audit timestamps must be timezone-aware")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"non-canonicalisable value in audit payload: {type(value)!r}")


def canonical_preimage(payload: dict[str, Any]) -> bytes:
    """Canonical byte serialisation of an audit payload (the dict must already
    exclude `hash`)."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_canonical_default,
    ).encode("utf-8")


def compute_hash(payload_without_hash: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical preimage."""
    return hashlib.sha256(canonical_preimage(payload_without_hash)).hexdigest()
