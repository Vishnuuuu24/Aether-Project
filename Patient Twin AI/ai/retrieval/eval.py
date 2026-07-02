"""Retrieval eval harness — Recall@K / MRR / nDCG (docs/10 T3.2 DoD; docs/11).

Runs a set of labelled queries through a `Retriever` and pools the ranking metrics.
Relevance is binary (a set of relevant chunk_ids per query).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from ai.interfaces.retriever import Retriever
from schemas.retrieval import RetrievalScope


@dataclass(frozen=True)
class EvalQuery:
    query: str
    scope: RetrievalScope
    relevant_ids: frozenset[UUID]


@dataclass(frozen=True)
class EvalResult:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    n_queries: int


def recall_at_k(retrieved: Sequence[UUID], relevant: frozenset[UUID], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: Sequence[UUID], relevant: frozenset[UUID]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[UUID], relevant: frozenset[UUID], k: int) -> float:
    dcg = 0.0
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(retriever: Retriever, queries: Sequence[EvalQuery], *, k: int = 10) -> EvalResult:
    if not queries:
        raise ValueError("no queries to evaluate")
    recalls, rrs, ndcgs = [], [], []
    for q in queries:
        retrieved = [c.chunk_id for c in retriever.search(q.query, q.scope, k=k)]
        recalls.append(recall_at_k(retrieved, q.relevant_ids, k))
        rrs.append(reciprocal_rank(retrieved, q.relevant_ids))
        ndcgs.append(ndcg_at_k(retrieved, q.relevant_ids, k))
    n = len(queries)
    return EvalResult(
        recall_at_k=sum(recalls) / n,
        mrr=sum(rrs) / n,
        ndcg_at_k=sum(ndcgs) / n,
        k=k,
        n_queries=n,
    )
