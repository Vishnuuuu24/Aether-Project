"""BM25 lexical scorer."""

from __future__ import annotations

from ai.retrieval.bm25 import BM25


def test_ranks_matching_document_highest() -> None:
    corpus = [
        "metformin is first-line therapy for type 2 diabetes",
        "atrial fibrillation requires anticoagulation",
        "hypertension is managed with lifestyle and medication",
    ]
    scores = BM25(corpus).scores("type 2 diabetes metformin")
    assert scores[0] == max(scores)
    assert scores[0] > 0.0


def test_non_matching_query_scores_zero() -> None:
    scores = BM25(["the quick brown fox"]).scores("pneumonia sepsis")
    assert scores == [0.0]


def test_empty_corpus() -> None:
    assert BM25([]).scores("anything") == []
