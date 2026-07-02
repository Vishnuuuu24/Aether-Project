"""InMemoryVectorStore cosine ordering; QdrantVectorStore integration (skip if down)."""

from __future__ import annotations

import pytest

from ai.retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore

from ._corpus import kb


def test_in_memory_ranks_by_cosine() -> None:
    store = InMemoryVectorStore()
    a, b = kb("a", index=0), kb("b", index=1)
    store.upsert([(a, [1.0, 0.0]), (b, [0.0, 1.0])])
    hits = store.search([1.0, 0.0], k=2)
    assert hits[0][0].chunk_id == a.chunk_id
    assert hits[0][1] > hits[1][1]


def test_qdrant_roundtrip_or_skip() -> None:
    a, b = kb("alpha", index=0), kb("beta", index=1)
    try:
        store = QdrantVectorStore(dim=2)
        store.upsert([(a, [1.0, 0.0]), (b, [0.0, 1.0])])
        hits = store.search([1.0, 0.0], k=2)
    except Exception as exc:  # noqa: BLE001 - any connection/setup failure -> skip
        pytest.skip(f"Qdrant not available: {exc}")
    assert hits[0][0].chunk_id == a.chunk_id
