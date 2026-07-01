"""Vector payload validation (Qdrant). Covers the KB-vs-patient invariants and
timezone/blank-text rules.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas import ConsentScope, VectorPayload, VectorSourceType

TS = datetime(2026, 6, 1, tzinfo=UTC)


def patient_chunk(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "source_type": VectorSourceType.PATIENT_RECORD,
        "patient_id": uuid4(),
        "source_document_id": uuid4(),
        "chunk_text": "Discharge summary: stable, follow up in 2 weeks.",
        "chunk_index": 0,
        "embedding_model": "medcpt",
        "consent_scope": ConsentScope.DOCUMENTS,
        "timestamp": TS,
    }
    data.update(overrides)
    return data


def test_valid_patient_record() -> None:
    v = VectorPayload(**patient_chunk())
    assert v.source_type is VectorSourceType.PATIENT_RECORD
    assert v.chunk_id is not None


def test_valid_kb_passage() -> None:
    v = VectorPayload(
        source_type=VectorSourceType.KB_PASSAGE,
        chunk_text="Resting tachycardia is a sustained HR above…",
        chunk_index=3,
        embedding_model="bge",
        timestamp=TS,
    )
    assert v.patient_id is None


def test_patient_record_requires_patient_id() -> None:
    with pytest.raises(ValidationError):
        VectorPayload(**patient_chunk(patient_id=None))


def test_kb_passage_must_not_carry_patient_id() -> None:
    with pytest.raises(ValidationError):
        VectorPayload(
            source_type=VectorSourceType.KB_PASSAGE,
            patient_id=uuid4(),
            chunk_text="x",
            chunk_index=0,
            embedding_model="bge",
            timestamp=TS,
        )


def test_naive_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        VectorPayload(**patient_chunk(timestamp=datetime(2026, 6, 1)))


def test_blank_text_and_negative_index_rejected() -> None:
    with pytest.raises(ValidationError):
        VectorPayload(**patient_chunk(chunk_text="   "))
    with pytest.raises(ValidationError):
        VectorPayload(**patient_chunk(chunk_index=-1))


def test_roundtrip_serialise_validate() -> None:
    v = VectorPayload(**patient_chunk(codes=["LOINC:8867-4"], section="Plan"))
    assert VectorPayload.model_validate_json(v.model_dump_json()) == v
