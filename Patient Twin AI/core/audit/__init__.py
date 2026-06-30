"""core.audit — hash-chained, append-only audit trail (docs/04 §7, docs/06 §8).

Every mutation in the system emits an audit record through `AuditWriter`, which
links it into a single global chain. `verify_chain` re-derives the chain to prove
no record was altered, inserted, or removed.

The SQLAlchemy store is intentionally NOT imported here (it pulls in SQLAlchemy);
import it explicitly from `core.audit.sql_store` where a DB session exists.
"""

from __future__ import annotations

from .chain import AuditWriter, Store, record_hash, verify_chain, verify_record
from .errors import AuditChainError
from .hashing import GENESIS_PREV_HASH, canonical_preimage, compute_hash
from .store import InMemoryAuditStore

__all__ = [
    "AuditWriter",
    "Store",
    "record_hash",
    "verify_chain",
    "verify_record",
    "AuditChainError",
    "GENESIS_PREV_HASH",
    "canonical_preimage",
    "compute_hash",
    "InMemoryAuditStore",
]
