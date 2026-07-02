"""Rerankers (docs/03 §2.5: cross-encoder over top-k). `LexicalReranker` is a
deterministic token-overlap scorer for the fast suite; `HfReranker` is the real
cross-encoder (bge-reranker-v2-m3), loaded lazily.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class LexicalReranker:
    """Jaccard token-overlap reranker. Dev/test only."""

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
