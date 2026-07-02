"""Vector stores (docs/03 §1: Qdrant). `InMemoryVectorStore` is a real cosine store
that drives the fast suite and the eval harness; `QdrantVectorStore` is the
production adapter, tested against the compose container and skipping when it is
unavailable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from schemas.vector import VectorPayload

_COLLECTION = "patient_copilot_chunks"


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._points: list[tuple[VectorPayload, list[float]]] = []

    def upsert(self, points: Sequence[tuple[VectorPayload, list[float]]]) -> None:
        self._points.extend(points)

    def search(self, vector: list[float], *, k: int) -> list[tuple[VectorPayload, float]]:
        scored = [(payload, _cosine(vector, vec)) for payload, vec in self._points]
        scored.sort(key=lambda pair: (pair[1], pair[0].chunk_id.int), reverse=True)
        return scored[:k]


class QdrantVectorStore:
    """Real Qdrant adapter (docs/08). Lazily connects; raises if the client or server
    is unavailable so integration tests skip cleanly.
    """

    def __init__(
        self, *, url: str | None = None, collection: str = _COLLECTION, dim: int | None = None
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError("qdrant-client is not installed") from exc
        self._client = QdrantClient(url=url or "http://localhost:6333")
        self._collection = collection
        if dim is not None:
            if self._client.collection_exists(collection):
                self._client.delete_collection(collection)
            self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(
        self, points: Sequence[tuple[VectorPayload, list[float]]]
    ) -> None:  # pragma: no cover - needs a running Qdrant
        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=str(payload.chunk_id),
                    vector=vec,
                    payload=payload.model_dump(mode="json"),
                )
                for payload, vec in points
            ],
        )

    def search(
        self, vector: list[float], *, k: int
    ) -> list[tuple[VectorPayload, float]]:  # pragma: no cover - needs a running Qdrant
        hits = self._client.search(
            collection_name=self._collection, query_vector=vector, limit=k
        )
        return [(VectorPayload.model_validate(hit.payload), float(hit.score)) for hit in hits]
