"""Stable `Retriever` interface (docs/02 §6).

v1 impl: `ai.retrieval.HybridRetriever` (BM25 + dense + cross-encoder rerank).
Deferred swaps (a different embedder/reranker/vector store) are new implementations
of the ports behind this retriever, never a new call site. Retrieval is always
consent-scoped (docs/06).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas.retrieval import EvidenceChunk, RetrievalScope


@runtime_checkable
class Retriever(Protocol):
    def search(self, query: str, scope: RetrievalScope, *, k: int = 10) -> list[EvidenceChunk]: ...
