"""Document-coding contracts (docs/04 §4; T3.1).

The pipeline is: `ClinicalDocument` → OCR → clinical coding → `CodedEntity[]`, each
gated by a confidence threshold into `proposed` / `committed` (sub-threshold stays
`proposed` and is never silently committed, docs/04 §4). The Patient State Engine
maps committed/proposed entities onto the PSG's Condition/Medication/Observation/
Allergy nodes.

CLAUDE.md: contracts live in schemas/ only.
"""

from __future__ import annotations

from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class DocumentType(str, Enum):
    DISCHARGE_SUMMARY = "discharge_summary"
    CLINICAL_NOTE = "clinical_note"
    LAB_REPORT = "lab_report"
    SOP = "sop"
    MEDICAL_TEXT = "medical_text"
    OTHER = "other"


class EntityType(str, Enum):
    CONDITION = "condition"  # -> SNOMED CT, ConditionNode
    MEDICATION = "medication"  # -> RxNorm, MedicationNode
    OBSERVATION = "observation"  # -> LOINC, ObservationNode
    ALLERGY = "allergy"  # -> SNOMED/RxNorm, AllergyNode


class CodeStatus(str, Enum):
    PROPOSED = "proposed"  # below confidence threshold — awaits human confirmation
    COMMITTED = "committed"  # at/above threshold — may enter the PSG as committed


class ClinicalDocument(BaseModel):
    """A document to ingest. Provide inline `text` (already-OCR'd / dev) or a `uri`
    the OCR engine reads.
    """

    patient_id: UUID
    doc_type: DocumentType
    document_id: UUID = Field(default_factory=uuid4)
    text: str | None = None
    uri: str | None = None

    @model_validator(mode="after")
    def _has_source(self) -> ClinicalDocument:
        if not (self.text and self.text.strip()) and not (self.uri and self.uri.strip()):
            raise ValueError("document must provide non-blank text or a uri")
        return self


class CodedEntity(BaseModel):
    """One coded finding from a document (docs/04 §4)."""

    entity_type: EntityType
    code_system: str  # "SNOMED CT" | "RxNorm" | "LOINC" — the terminology, set by the coder
    code: str
    display: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: CodeStatus = CodeStatus.PROPOSED
    value: str | None = None  # observation value
    unit: str | None = None  # observation unit
    section: str | None = None  # source section/heading, if known


class DocumentCodingResult(BaseModel):
    document_id: UUID
    patient_id: UUID
    doc_type: DocumentType
    entities: list[CodedEntity] = Field(default_factory=list)
    coder_version: str
