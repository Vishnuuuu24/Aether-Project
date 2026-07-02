"""Retrieval contracts (docs/02 §6, docs/10 T3.2).

`RetrievalScope` carries the consent context a search runs under; `EvidenceChunk` is
a single retrieved passage with its score. Retrieval spans two sources — the global
clinical KB (`kb_passage`) and the patient's own record (`patient_record`) — and is
consent-scoped: patient content is only ever returned for the matching patient under
a consented scope (docs/06).

CLAUDE.md: contracts live in schemas/ only.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from .consent import ConsentScope
from .vector import VectorSourceType


class RetrievalScope(BaseModel):
    """The consent context for a query. `patient_id` bounds which patient records may
    be searched; `consented_scopes` are the scopes in force; `include_kb` allows the
    global KB (never patient-specific).
    """

    patient_id: UUID | None = None
    consented_scopes: list[ConsentScope] = Field(default_factory=list)
    include_kb: bool = True


class EvidenceChunk(BaseModel):
    chunk_id: UUID
    source_type: VectorSourceType
    text: str
    score: float
    patient_id: UUID | None = None
    source_document_id: UUID | None = None
    codes: list[str] = Field(default_factory=list)
    section: str | None = None
