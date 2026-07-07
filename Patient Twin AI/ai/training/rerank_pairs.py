"""Cross-encoder training-pair builder from nfcorpus qrels (docs/16 Sprint 11).

Turns an `IrSplit` (real query→doc relevance judgements) into labelled
`(query, passage, label)` examples for fine-tuning the cross-encoder reranker.

- **Positives** come straight from the qrels (score >= 1) — never fabricated.
- **Negatives** are mined with BM25 (`ai/retrieval/bm25.py`): the highest-BM25 docs
  that are NOT judged relevant for that query. Hard negatives (lexically similar but
  irrelevant) are the standard, informative signal for cross-encoder training; random
  negatives are too easy. Any qrels-judged doc is excluded from a query's negatives so
  a graded-but-relevant doc is never mislabelled 0.

Deterministic given a seed (BM25 is deterministic; ties broken by doc-id). The output
is a plain list of `RerankExample`, framework-agnostic — the trainer wraps them into
whatever the cross-encoder library expects.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ai.eval_datasets.nfcorpus import IrSplit
from ai.retrieval.bm25 import BM25


@dataclass(frozen=True)
class RerankExample:
    """One labelled pair: `label` is 1.0 (relevant) or 0.0 (mined negative)."""

    query: str
    passage: str
    label: float


def build_rerank_examples(
    split: IrSplit,
    *,
    negatives_per_positive: int = 4,
    max_positives_per_query: int | None = None,
    max_bm25_pool: int = 100,
    seed: int = 0,
) -> list[RerankExample]:
    """Build labelled pairs: qrels positives + `negatives_per_positive` BM25 hard
    negatives per positive, drawn from the top-`max_bm25_pool` BM25 docs that are not
    judged for the query. Falls back to random unjudged docs if BM25 yields too few.

    `max_positives_per_query` caps how many positives a query contributes, keeping the
    ones BM25 ranks HIGHEST (the docs that actually reach the top of the candidate pool
    at inference, so the most informative to reorder). This is a *quality* control: a
    handful of nfcorpus queries mark hundreds–1363 of the 3633 docs "relevant", which is
    near-random supervision that dilutes ranking signal — not a speed cap on the useful
    data. `None` keeps every positive.
    """
    if negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be >= 0")
    doc_ids = list(split.corpus.keys())
    bm25 = BM25([split.corpus[d] for d in doc_ids])
    rng = random.Random(seed)

    examples: list[RerankExample] = []
    for qid in split.query_ids:
        query = split.queries[qid]
        judged = set(split.qrels.get(qid, {}))  # any judged doc (incl. graded) — never a negative
        relevant = split.relevant_ids(qid)
        if not relevant:
            continue

        # BM25-ranked candidate pool for this query, minus judged docs → hard negatives.
        scores = bm25.scores(query)
        ranked = sorted(range(len(doc_ids)), key=lambda i: (scores[i], doc_ids[i]), reverse=True)

        # Positives, ordered by BM25 so a cap keeps the top (pool-reaching) ones.
        positives = [doc_ids[i] for i in ranked if doc_ids[i] in relevant]
        if max_positives_per_query is not None:
            positives = positives[:max_positives_per_query]

        hard_negs = [doc_ids[i] for i in ranked[:max_bm25_pool] if doc_ids[i] not in judged]

        n_needed = negatives_per_positive * len(positives)
        negs = hard_negs[:n_needed]
        if len(negs) < n_needed:  # pad with random unjudged docs (small corpora / broad qrels)
            pool = [d for d in doc_ids if d not in judged and d not in set(negs)]
            rng.shuffle(pool)
            negs += pool[: n_needed - len(negs)]

        for docid in positives:
            examples.append(RerankExample(query, split.corpus[docid], 1.0))
        for docid in negs:
            examples.append(RerankExample(query, split.corpus[docid], 0.0))

    rng.shuffle(examples)
    return examples
