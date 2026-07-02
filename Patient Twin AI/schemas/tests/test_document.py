"""Contract tests for schemas/document.py."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas.document import (
    ClinicalDocument,
    CodedEntity,
    CodeStatus,
    DocumentType,
    EntityType,
)


def test_document_requires_text_or_uri() -> None:
    with pytest.raises(ValidationError, match="text or a uri"):
        ClinicalDocument(patient_id=uuid4(), doc_type=DocumentType.DISCHARGE_SUMMARY)


def test_document_accepts_inline_text() -> None:
    doc = ClinicalDocument(
        patient_id=uuid4(), doc_type=DocumentType.DISCHARGE_SUMMARY, text="Diagnosis: ..."
    )
    assert doc.text is not None


def test_coded_entity_defaults_to_proposed() -> None:
    entity = CodedEntity(
        entity_type=EntityType.CONDITION,
        code_system="SNOMED CT",
        code="44054006",
        display="Type 2 diabetes mellitus",
        confidence=0.9,
    )
    assert entity.status is CodeStatus.PROPOSED


def test_coded_entity_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        CodedEntity(
            entity_type=EntityType.CONDITION,
            code_system="SNOMED CT",
            code="x",
            display="x",
            confidence=1.5,
        )
