"""Vector-store payload contract (Qdrant). The metadata attached to every
embedding for hybrid retrieval (BM25 + dense + rerank; docs/02, docs/10 T3.2).

This is the *payload*, not the vector: the embedding array is stored by Qdrant
and is never part of this contract. Retrieval is consent-scoped and spans two
sources — the global clinical KB and the patient's own record. `source_type`
distinguishes them and enforces a PHI-hygiene invariant: KB content is global and
must not be bound to a `patient_id`, and patient content must carry one.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from .consent import ConsentScope


class VectorSourceType(str, Enum):
    KB_PASSAGE = "kb_passage"  # global clinical knowledge base (not patient-specific)
    PATIENT_RECORD = "patient_record"  # chunk derived from this patient's documents/record


class VectorPayload(BaseModel):
    """A Qdrant point's payload. Filterable/returnable metadata for one chunk.

    `source_type` mirrors the output contract's `EvidenceKind` so a retrieved KB
    chunk maps cleanly to `kb_passage` evidence (docs/04 §6).
    """

    chunk_id: UUID = Field(default_factory=uuid4)
    source_type: VectorSourceType
    # Required for PATIENT_RECORD; MUST be None for KB_PASSAGE (see validator below).
    patient_id: UUID | None = None
    source_document_id: UUID | None = None
    chunk_text: str
    chunk_index: int = Field(ge=0)  # position of the chunk within its source document
    embedding_model: str  # which embedder produced the vector (e.g. medcpt, bge) — hybrid needs it
    consent_scope: ConsentScope | None = None  # scope this chunk falls under (None for KB)
    codes: list[str] = Field(default_factory=list)  # LOINC/SNOMED/RxNorm tags, if coded
    section: str | None = None  # source section/heading, if known
    timestamp: datetime  # ingest/source time (RFC 3339 with timezone)

    @field_validator("timestamp")
    @classmethod
    def _timestamp_tz(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must include timezone — naive datetimes are rejected")
        return v

    @field_validator("chunk_text")
    @classmethod
    def _text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk_text must not be blank")
        return v

    @model_validator(mode="after")
    def _source_invariants(self) -> VectorPayload:
        if self.source_type is VectorSourceType.PATIENT_RECORD and self.patient_id is None:
            raise ValueError("patient_record vectors require a patient_id")
        if self.source_type is VectorSourceType.KB_PASSAGE and self.patient_id is not None:
            raise ValueError("kb_passage vectors are global and must not carry a patient_id")
        return self
