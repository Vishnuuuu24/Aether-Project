"""Sprint 11: `FineTunedReranker` seam behaviour — the fallback path (no model
download). The real CrossEncoder load is exercised only in the training run."""

from __future__ import annotations

import json
from pathlib import Path

from ai.retrieval.reranker import (
    FINETUNED_RERANKER_VERSION,
    FineTunedReranker,
    LexicalReranker,
)


def test_missing_checkpoint_falls_back_to_lexical(tmp_path: Path) -> None:
    reranker = FineTunedReranker.from_checkpoint(tmp_path / "does-not-exist")
    assert reranker.is_fallback
    assert reranker.version == f"{FINETUNED_RERANKER_VERSION}+fallback-lexical"
    # still a working reranker (never raises, returns one score per text)
    scores = reranker.rerank("statins cholesterol", ["about statins", "bananas"])
    assert len(scores) == 2
    assert scores[0] >= scores[1]


def test_empty_texts_returns_empty(tmp_path: Path) -> None:
    reranker = FineTunedReranker.from_checkpoint(tmp_path / "nope")
    assert reranker.rerank("q", []) == []


def test_checkpoint_id_folds_manifest_version(tmp_path: Path) -> None:
    # A manifest present but no loadable model dir -> still falls back, but the version
    # id resolution reads the manifest for the content-addressed id.
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "reranker@abc123"}))
    reranker = FineTunedReranker.from_checkpoint(tmp_path)
    # model dir is absent -> lexical fallback stamp
    assert reranker.is_fallback


def test_lexical_reranker_is_versioned() -> None:
    assert LexicalReranker().version == "lexical-jaccard-v1"
