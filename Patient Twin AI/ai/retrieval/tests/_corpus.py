"""Seed corpus builders for retrieval tests."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from schemas.consent import ConsentScope
from schemas.vector import VectorPayload, VectorSourceType

_TS = datetime(2026, 1, 1, tzinfo=UTC)


def kb(text: str, *, index: int = 0) -> VectorPayload:
    return VectorPayload(
        source_type=VectorSourceType.KB_PASSAGE,
        chunk_text=text,
        chunk_index=index,
        embedding_model="hash-dev",
        timestamp=_TS,
    )


def patient_chunk(
    patient_id: UUID,
    text: str,
    *,
    scope: ConsentScope = ConsentScope.DOCUMENTS,
    index: int = 0,
) -> VectorPayload:
    return VectorPayload(
        source_type=VectorSourceType.PATIENT_RECORD,
        patient_id=patient_id,
        chunk_text=text,
        chunk_index=index,
        embedding_model="hash-dev",
        consent_scope=scope,
        timestamp=_TS,
    )
