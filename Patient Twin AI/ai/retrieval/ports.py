"""Retrieval sub-component ports (docs/02 §6, docs/03 §2.4-2.5).

Swappable model backends behind the `Retriever` interface: `Embedder` (MedCPT/BGE),
`Reranker` (cross-encoder), `VectorStore` (Qdrant). Real adapters load their weights
lazily; deterministic fakes drive the fast test suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from schemas.vector import VectorPayload


@runtime_checkable
class Embedder(Protocol):
    model_name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, texts: Sequence[str]) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, points: Sequence[tuple[VectorPayload, list[float]]]) -> None: ...

    def search(
        self, vector: list[float], *, k: int
    ) -> list[tuple[VectorPayload, float]]: ...
