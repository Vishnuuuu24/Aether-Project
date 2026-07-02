"""Contract tests for schemas/retrieval.py."""

from __future__ import annotations

from uuid import uuid4

from schemas.retrieval import EvidenceChunk, RetrievalScope
from schemas.vector import VectorSourceType


def test_scope_defaults() -> None:
    scope = RetrievalScope()
    assert scope.include_kb is True
    assert scope.patient_id is None
    assert scope.consented_scopes == []


def test_evidence_chunk_roundtrip() -> None:
    chunk = EvidenceChunk(
        chunk_id=uuid4(),
        source_type=VectorSourceType.KB_PASSAGE,
        text="metformin is first-line for type 2 diabetes",
        score=1.23,
    )
    assert EvidenceChunk.model_validate_json(chunk.model_dump_json()) == chunk
