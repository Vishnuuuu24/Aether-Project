"""Real embedder/reranker adapters — exercised only when the ML stack (requirements-
ml.txt) is installed; skipped otherwise so the base suite stays light.
"""

from __future__ import annotations

import pytest


def test_hf_embedder_produces_vectors() -> None:
    pytest.importorskip("sentence_transformers", reason="ML stack not installed")
    from ai.retrieval.embedder import HfEmbedder

    embedder = HfEmbedder("BAAI/bge-small-en-v1.5")  # small model keeps the test cheap
    vectors = embedder.embed(["type 2 diabetes management", "atrial fibrillation"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert all(isinstance(x, float) for x in vectors[0])


def test_hf_reranker_scores_candidates() -> None:
    pytest.importorskip("sentence_transformers", reason="ML stack not installed")
    from ai.retrieval.reranker import HfReranker

    reranker = HfReranker("BAAI/bge-reranker-base")  # base is lighter than v2-m3
    scores = reranker.rerank(
        "diabetes treatment",
        ["metformin for type 2 diabetes", "anticoagulation for atrial fibrillation"],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]  # the diabetes passage is more relevant
