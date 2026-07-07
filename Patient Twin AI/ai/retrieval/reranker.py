"""Rerankers (docs/03 §2.5: cross-encoder over top-k). `LexicalReranker` is a
deterministic token-overlap scorer for the fast suite; `HfReranker` is the real
cross-encoder (bge-reranker-v2-m3), loaded lazily; `FineTunedReranker` loads a
Sprint-11 fine-tuned checkpoint and degrades to `LexicalReranker` when the ML stack
or the checkpoint is unavailable (never raises — the seam always has a fallback).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")

# Base identity for the Sprint 11 fine-tuned reranker; `from_checkpoint` folds the
# content-addressed checkpoint id into it (docs/04 §7 versioning/audit).
FINETUNED_RERANKER_VERSION = "bge-reranker-v2-m3-nfcorpus-v1"
LEXICAL_RERANKER_VERSION = "lexical-jaccard-v1"


class LexicalReranker:
    """Jaccard token-overlap reranker. Dev/test only."""

    version = LEXICAL_RERANKER_VERSION

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        q = set(_TOKEN.findall(query.lower()))
        scores: list[float] = []
        for text in texts:
            t = set(_TOKEN.findall(text.lower()))
            union = q | t
            scores.append(len(q & t) / len(union) if union else 0.0)
        return scores


class HfReranker:
    """Real cross-encoder reranker (bge-reranker-v2-m3), loaded lazily. Raises
    ImportError if the ML stack is absent (integration tests skip).
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - needs sentence-transformers
            raise ImportError(
                "sentence-transformers is not installed; install requirements-ml.txt"
            ) from exc
        self._model = CrossEncoder(model_name)  # pragma: no cover - downloads weights
        self.model_name = model_name

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:  # pragma: no cover
        if not texts:
            return []
        scores = self._model.predict([(query, t) for t in texts])
        return [float(s) for s in scores]


class FineTunedReranker:
    """Sprint 11 fine-tuned cross-encoder wrapped as a `Reranker` (docs/16 Sprint 11).

    A new *implementation* of the reranker seam — never a new call site: it satisfies
    the same `Reranker` protocol `HybridRetriever` already consumes. Loaded from a
    fine-tuned checkpoint dir written by `ai/training/train_reranker.py`. Version is
    content-addressed (`{base}+{checkpoint-id}`) for audit stamping.

    `from_checkpoint` NEVER raises: if `sentence-transformers`/`torch` is absent, the
    checkpoint is missing, or loading fails, it degrades to `LexicalReranker` so the
    retrieval path always has a deterministic fallback (the classical/lexical reranker
    stays the fallback — DoD).
    """

    def __init__(self, model: object, *, version: str) -> None:
        self._model = model  # a sentence_transformers.CrossEncoder or a LexicalReranker
        self.version = version

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: Path, *, base_version: str = FINETUNED_RERANKER_VERSION
    ) -> FineTunedReranker:
        """Load the fine-tuned CrossEncoder from `<checkpoint_dir>/model/`, folding the
        checkpoint id into the version. Falls back to lexical on any failure."""
        model_dir = checkpoint_dir / "model"
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(str(model_dir))
            stamped = f"{base_version}+{_checkpoint_id(checkpoint_dir)}"
            return cls(model, version=stamped)
        except Exception:  # noqa: BLE001 - any load failure must degrade, never crash the seam
            return cls(LexicalReranker(), version=f"{base_version}+fallback-lexical")

    @property
    def is_fallback(self) -> bool:
        return isinstance(self._model, LexicalReranker)

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        if isinstance(self._model, LexicalReranker):
            return self._model.rerank(query, texts)
        scores = self._model.predict([(query, t) for t in texts])  # type: ignore[attr-defined]
        return [float(s) for s in scores]


def _checkpoint_id(checkpoint_dir: Path) -> str:
    """The content-addressed version from the checkpoint manifest, else the dir name."""
    manifest = checkpoint_dir / "manifest.json"
    if manifest.is_file():
        try:
            return str(json.loads(manifest.read_text()).get("version", checkpoint_dir.name))
        except (OSError, ValueError):
            return checkpoint_dir.name
    return checkpoint_dir.name
