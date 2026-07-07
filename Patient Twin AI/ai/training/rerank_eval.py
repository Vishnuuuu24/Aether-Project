"""Reranker A/B lift harness on a labelled IR split (docs/16 Sprint 11).

Measures the lift a cross-encoder gives over a first-stage lexical ranking, using the
**existing** retrieval metric functions (`ai/retrieval/eval.py`: `recall_at_k`,
`reciprocal_rank`, `ndcg_at_k`) so Sprint 11's number is comparable to the rest of the
retrieval eval. Relevance is binary (qrels score >= 1), matching those functions.

Protocol (mirrors `HybridRetriever`'s rerank step in isolation, so the measured lift is
attributable to the cross-encoder alone, not to changes in the candidate generator):

  1. First stage: BM25 over the corpus → top-`pool_size` candidate docs per query.
  2. Baseline arm: score the candidates in BM25 order (no cross-encoder).
  3. Reranked arm: reorder the SAME candidates by `reranker.rerank(query, texts)`.
  4. Report Recall@k / MRR / nDCG@k for both arms + the delta.

The candidate pool caps achievable recall (a doc BM25 never surfaces can't be reranked
in) — so recall is reported for context, but the reranker's job is ordering quality,
best read off MRR / nDCG@k. Ids stay BEIR strings; the metric funcs are id-type-agnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai.eval_datasets.nfcorpus import IrSplit
from ai.retrieval.bm25 import BM25
from ai.retrieval.eval import ndcg_at_k, recall_at_k, reciprocal_rank
from ai.retrieval.ports import Reranker


@dataclass(frozen=True)
class ArmMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int
    n_queries: int


@dataclass(frozen=True)
class LiftResult:
    baseline: ArmMetrics  # BM25-order candidates (no cross-encoder)
    reranked: ArmMetrics  # same candidates, cross-encoder order
    pool_size: int

    @property
    def ndcg_lift(self) -> float:
        return self.reranked.ndcg_at_k - self.baseline.ndcg_at_k

    @property
    def mrr_lift(self) -> float:
        return self.reranked.mrr - self.baseline.mrr


def _pool_metrics(
    ordered_ids_per_query: list[Sequence[str]],
    relevant_per_query: list[frozenset[str]],
    *,
    k: int,
) -> ArmMetrics:
    recalls, rrs, ndcgs = [], [], []
    for retrieved, relevant in zip(ordered_ids_per_query, relevant_per_query, strict=True):
        recalls.append(recall_at_k(retrieved, relevant, k))  # type: ignore[arg-type]
        rrs.append(reciprocal_rank(retrieved, relevant))  # type: ignore[arg-type]
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))  # type: ignore[arg-type]
    n = len(ordered_ids_per_query)
    return ArmMetrics(
        recall_at_k=sum(recalls) / n,
        mrr=sum(rrs) / n,
        ndcg_at_k=sum(ndcgs) / n,
        k=k,
        n_queries=n,
    )


def evaluate_rerank_lift(
    split: IrSplit,
    reranker: Reranker,
    *,
    k: int = 10,
    pool_size: int = 100,
) -> LiftResult:
    """A/B a reranker vs BM25-only on `split`. Returns both arms + the lift."""
    if not split.query_ids:
        raise ValueError("no queries to evaluate")
    doc_ids = list(split.corpus.keys())
    bm25 = BM25([split.corpus[d] for d in doc_ids])

    baseline_orders: list[Sequence[str]] = []
    reranked_orders: list[Sequence[str]] = []
    relevants: list[frozenset[str]] = []

    for qid in split.query_ids:
        query = split.queries[qid]
        scores = bm25.scores(query)
        ranked = sorted(range(len(doc_ids)), key=lambda i: (scores[i], doc_ids[i]), reverse=True)
        pool_ids = [doc_ids[i] for i in ranked[:pool_size]]

        baseline_orders.append(pool_ids)
        rr_scores = reranker.rerank(query, [split.corpus[d] for d in pool_ids])
        ranked_pool = sorted(
            zip(pool_ids, rr_scores, strict=True), key=lambda x: x[1], reverse=True
        )
        reordered = [d for d, _ in ranked_pool]
        reranked_orders.append(reordered)
        relevants.append(split.relevant_ids(qid))

    return LiftResult(
        baseline=_pool_metrics(baseline_orders, relevants, k=k),
        reranked=_pool_metrics(reranked_orders, relevants, k=k),
        pool_size=pool_size,
    )
