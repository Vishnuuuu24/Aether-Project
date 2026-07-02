"""Okapi BM25 lexical scorer (docs/10 T3.2 — the lexical leg of hybrid retrieval).

Pure-Python and dependency-free so the base suite has a real BM25 without pulling
`rank_bm25`. Standard BM25 with k1=1.5, b=0.75.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, corpus: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs = [_tokenize(doc) for doc in corpus]
        self._doc_len = [len(d) for d in self._docs]
        n = len(self._docs)
        self._avg_len = (sum(self._doc_len) / n) if n else 0.0
        self._freqs = [Counter(d) for d in self._docs]
        # Document frequency per term.
        df: Counter[str] = Counter()
        for freq in self._freqs:
            df.update(freq.keys())
        # BM25+ style idf, floored at 0 so common terms never contribute negatively.
        self._idf = {
            term: max(0.0, math.log((n - d + 0.5) / (d + 0.5) + 1.0)) for term, d in df.items()
        }

    def scores(self, query: str) -> list[float]:
        q_terms = _tokenize(query)
        out: list[float] = []
        for i, freq in enumerate(self._freqs):
            score = 0.0
            norm = self._k1 * (1.0 - self._b + self._b * self._doc_len[i] / (self._avg_len or 1.0))
            for term in q_terms:
                if term not in freq:
                    continue
                tf = freq[term]
                score += self._idf.get(term, 0.0) * (tf * (self._k1 + 1.0)) / (tf + norm)
            out.append(score)
        return out
