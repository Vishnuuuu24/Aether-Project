"""Embedders (docs/03 §2.4: MedCPT primary, BGE second view).

`HashEmbedder` is a deterministic hashing bag-of-words embedder — no weights, real
cosine behaviour (similar text → similar vectors) — for the fast suite. `HfEmbedder`
is the real adapter (MedCPT / BGE via sentence-transformers), loaded lazily.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


class HashEmbedder:
    """Deterministic hashed bag-of-words vectors. Dev/test only, not semantic."""

    def __init__(self, *, dim: int = 64, model_name: str = "hash-dev") -> None:
        self._dim = dim
        self.model_name = model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.sha1(token.encode()).digest()  # noqa: S324 - non-crypto hashing
            idx = digest[0] % self._dim
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec


class HfEmbedder:
    """Real embedder (MedCPT / BGE) via sentence-transformers, loaded lazily.
    Raises ImportError if the ML stack is not installed (integration tests skip).
    """

    def __init__(self, model_name: str = "ncbi/MedCPT-Query-Encoder") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - needs sentence-transformers
            raise ImportError(
                "sentence-transformers is not installed; install requirements-ml.txt"
            ) from exc
        self._model = SentenceTransformer(model_name)  # pragma: no cover - downloads weights
        self.model_name = model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - needs weights
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]
