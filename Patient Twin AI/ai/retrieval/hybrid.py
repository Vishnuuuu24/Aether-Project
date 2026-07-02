"""HybridRetriever — BM25 + dense + cross-encoder rerank (docs/10 T3.2).

Indexes a corpus at construction (BM25 over the text; embeddings upserted to the
vector store). A query fuses the lexical and dense candidate rankings with
Reciprocal Rank Fusion, reranks the fused top-N with the cross-encoder, and returns
the top-k. **Consent scoping is enforced on every candidate** before it can surface:
KB passages need `include_kb`; patient records need the matching patient_id AND a
consented scope (docs/06).
"""

from __future__ import annotations

from schemas.retrieval import EvidenceChunk, RetrievalScope
from schemas.vector import VectorPayload, VectorSourceType

from .bm25 import BM25
from .ports import Embedder, Reranker, VectorStore

RETRIEVER_VERSION = "hybrid-v1"
_RRF_K = 60  # standard Reciprocal Rank Fusion constant


class HybridRetriever:
    """Implements the `Retriever` protocol (docs/02 §6)."""

    def __init__(
        self,
        corpus: list[VectorPayload],
        *,
        embedder: Embedder,
        reranker: Reranker,
        vector_store: VectorStore,
        rerank_candidates: int = 20,
        version: str = RETRIEVER_VERSION,
    ) -> None:
        self._corpus = list(corpus)
        self._embedder = embedder
        self._reranker = reranker
        self._store = vector_store
        self._rerank_n = rerank_candidates
        self._version = version
        self._bm25 = BM25([c.chunk_text for c in self._corpus])
        if self._corpus:
            vectors = embedder.embed([c.chunk_text for c in self._corpus])
            vector_store.upsert(list(zip(self._corpus, vectors, strict=True)))

    def search(self, query: str, scope: RetrievalScope, *, k: int = 10) -> list[EvidenceChunk]:
        if not self._corpus:
            return []
        pool = max(self._rerank_n, k)

        # Dense ranking (only over what this scope may see).
        query_vec = self._embedder.embed([query])[0]
        dense = [
            payload
            for payload, _ in self._store.search(query_vec, k=pool * 2)
            if _is_visible(payload, scope)
        ][:pool]

        # Lexical ranking (BM25 over the corpus), consent-filtered.
        lex_scores = self._bm25.scores(query)
        lex_ranked = sorted(
            range(len(self._corpus)), key=lambda i: lex_scores[i], reverse=True
        )
        lexical = [
            self._corpus[i]
            for i in lex_ranked
            if lex_scores[i] > 0.0 and _is_visible(self._corpus[i], scope)
        ][:pool]

        fused = _rrf_fuse(dense, lexical)[:pool]
        if not fused:
            return []

        # Cross-encoder rerank the fused candidates.
        rerank_scores = self._reranker.rerank(query, [p.chunk_text for p in fused])
        ranked = sorted(zip(fused, rerank_scores, strict=True), key=lambda x: x[1], reverse=True)
        return [_to_chunk(payload, score) for payload, score in ranked[:k]]


def _is_visible(payload: VectorPayload, scope: RetrievalScope) -> bool:
    """Consent gate for a single chunk (docs/06)."""
    if payload.source_type is VectorSourceType.KB_PASSAGE:
        return scope.include_kb
    # PATIENT_RECORD: must match the patient AND fall under a consented scope.
    if payload.patient_id is None or payload.patient_id != scope.patient_id:
        return False
    return payload.consent_scope is not None and payload.consent_scope in scope.consented_scopes


def _rrf_fuse(*rankings: list[VectorPayload]) -> list[VectorPayload]:
    scores: dict[str, float] = {}
    seen: dict[str, VectorPayload] = {}
    for ranking in rankings:
        for rank, payload in enumerate(ranking):
            key = str(payload.chunk_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            seen.setdefault(key, payload)
    order = sorted(seen.keys(), key=lambda kk: (scores[kk], kk), reverse=True)
    return [seen[kk] for kk in order]


def _to_chunk(payload: VectorPayload, score: float) -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id=payload.chunk_id,
        source_type=payload.source_type,
        text=payload.chunk_text,
        score=float(score),
        patient_id=payload.patient_id,
        source_document_id=payload.source_document_id,
        codes=payload.codes,
        section=payload.section,
    )
