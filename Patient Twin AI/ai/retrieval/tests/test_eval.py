"""Recall@K / MRR / nDCG metrics + the harness running on a seed corpus (T3.2 DoD)."""

from __future__ import annotations

from math import log2
from uuid import uuid4

import pytest

from ai.retrieval.embedder import HashEmbedder
from ai.retrieval.eval import (
    EvalQuery,
    evaluate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ai.retrieval.hybrid import HybridRetriever
from ai.retrieval.reranker import LexicalReranker
from ai.retrieval.vector_store import InMemoryVectorStore
from schemas.retrieval import RetrievalScope

from ._corpus import kb


def test_metric_known_values() -> None:
    ids = [uuid4() for _ in range(3)]
    relevant = frozenset({ids[1]})  # relevant item at rank 2
    assert recall_at_k(ids, relevant, 3) == 1.0
    assert reciprocal_rank(ids, relevant) == 0.5
    assert ndcg_at_k(ids, relevant, 3) == pytest.approx(1.0 / log2(3))


def test_recall_zero_when_missing() -> None:
    ids = [uuid4(), uuid4()]
    assert recall_at_k(ids, frozenset({uuid4()}), 2) == 0.0
    assert reciprocal_rank(ids, frozenset({uuid4()})) == 0.0


def test_harness_runs_on_seed_corpus() -> None:
    corpus = [
        kb("metformin is first-line therapy for type 2 diabetes", index=0),
        kb("atrial fibrillation requires anticoagulation", index=1),
        kb("hypertension managed with lifestyle and medication", index=2),
    ]
    retriever = HybridRetriever(
        corpus,
        embedder=HashEmbedder(),
        reranker=LexicalReranker(),
        vector_store=InMemoryVectorStore(),
    )
    queries = [
        EvalQuery(
            query="type 2 diabetes treatment",
            scope=RetrievalScope(include_kb=True),
            relevant_ids=frozenset({corpus[0].chunk_id}),
        ),
        EvalQuery(
            query="anticoagulation for atrial fibrillation",
            scope=RetrievalScope(include_kb=True),
            relevant_ids=frozenset({corpus[1].chunk_id}),
        ),
    ]
    result = evaluate(retriever, queries, k=3)

    assert result.n_queries == 2
    assert 0.0 <= result.recall_at_k <= 1.0
    assert 0.0 <= result.mrr <= 1.0
    assert 0.0 <= result.ndcg_at_k <= 1.0
    assert result.recall_at_k > 0.0  # the relevant passages are retrievable


def test_evaluate_rejects_empty() -> None:
    retriever = HybridRetriever(
        [], embedder=HashEmbedder(), reranker=LexicalReranker(), vector_store=InMemoryVectorStore()
    )
    with pytest.raises(ValueError, match="no queries"):
        evaluate(retriever, [], k=3)
