"""Sprint 11: rerank pair-builder + lift-eval on a deterministic tiny corpus.

No model downloads — a synthetic `IrSplit` and fake rerankers exercise the plumbing
(labels, hard-negative exclusion, lift accounting) so the logic is covered in the fast
suite; the real cross-encoder is exercised only in the (skip-guarded) training run.
"""

from __future__ import annotations

from collections.abc import Sequence

from ai.eval_datasets.nfcorpus import IrSplit
from ai.retrieval.reranker import LexicalReranker
from ai.training.rerank_eval import evaluate_rerank_lift
from ai.training.rerank_pairs import build_rerank_examples


def _toy_split() -> IrSplit:
    corpus = {
        "D1": "statins lower cholesterol in the blood",
        "D2": "aspirin is an antiplatelet drug for the heart",
        "D3": "cholesterol is a lipid molecule",
        "D4": "bananas are a yellow fruit",
        "D5": "the heart pumps blood around the body",
        "D6": "unrelated text about mountains and rivers",
    }
    queries = {"Q1": "do statins lower cholesterol", "Q2": "aspirin for the heart"}
    qrels = {"Q1": {"D1": 2, "D3": 1}, "Q2": {"D2": 2, "D5": 1}}
    return IrSplit(split="test", queries=queries, corpus=corpus, qrels=qrels)


class _PerfectReranker:
    """Ranks known-relevant passages first — an oracle to prove lift is measured."""

    def __init__(self, split: IrSplit) -> None:
        self._relevant_texts = {
            split.corpus[d] for q in split.query_ids for d in split.relevant_ids(q)
        }

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        return [1.0 if t in self._relevant_texts else 0.0 for t in texts]


def test_pairs_positives_and_negatives_labelled() -> None:
    split = _toy_split()
    examples = build_rerank_examples(split, negatives_per_positive=2, seed=1)
    pos = [e for e in examples if e.label == 1.0]
    neg = [e for e in examples if e.label == 0.0]
    # 4 positives total (2 per query), each drawing 2 negatives -> 8 negatives.
    assert len(pos) == 4
    assert len(neg) == 8
    # positive passages are exactly the qrels-relevant docs
    rel_texts = {split.corpus[d] for q in split.query_ids for d in split.relevant_ids(q)}
    assert {e.passage for e in pos} == rel_texts


def test_pairs_never_use_a_judged_doc_as_negative() -> None:
    # Exclusion is per-query: a doc judged for Q1 may still be a valid negative for Q2
    # (where it is unjudged). So the invariant is "no negative is judged for ITS OWN
    # query", checked by query text.
    split = _toy_split()
    text_to_qid = {split.queries[q]: q for q in split.query_ids}
    judged_texts_by_qid = {
        q: {split.corpus[d] for d in split.qrels[q]} for q in split.query_ids
    }
    examples = build_rerank_examples(split, negatives_per_positive=3, seed=2)
    for e in examples:
        if e.label == 0.0:
            assert e.passage not in judged_texts_by_qid[text_to_qid[e.query]]


def test_pairs_positive_cap_limits_per_query() -> None:
    split = _toy_split()  # each query has 2 positives
    examples = build_rerank_examples(
        split, negatives_per_positive=2, max_positives_per_query=1, seed=3
    )
    pos = [e for e in examples if e.label == 1.0]
    # capped to 1 positive per query -> 2 positives total, each with 2 negatives.
    assert len(pos) == 2
    assert len([e for e in examples if e.label == 0.0]) == 4


def test_pairs_deterministic_given_seed() -> None:
    split = _toy_split()
    a = build_rerank_examples(split, negatives_per_positive=2, seed=7)
    b = build_rerank_examples(split, negatives_per_positive=2, seed=7)
    assert [(e.query, e.passage, e.label) for e in a] == [
        (e.query, e.passage, e.label) for e in b
    ]


def test_perfect_reranker_lifts_ndcg_and_mrr() -> None:
    split = _toy_split()
    result = evaluate_rerank_lift(split, _PerfectReranker(split), k=5, pool_size=6)
    assert result.reranked.ndcg_at_k >= result.baseline.ndcg_at_k
    assert result.reranked.ndcg_at_k > 0.0
    assert result.reranked.mrr == 1.0  # a relevant doc is ranked first for every query
    assert result.reranked.n_queries == 2


def test_lexical_reranker_runs_end_to_end() -> None:
    split = _toy_split()
    result = evaluate_rerank_lift(split, LexicalReranker(), k=5, pool_size=6)
    # Deterministic, finite metrics in [0, 1]; lift may be positive or ~0 on a toy set.
    assert 0.0 <= result.baseline.ndcg_at_k <= 1.0
    assert 0.0 <= result.reranked.ndcg_at_k <= 1.0
