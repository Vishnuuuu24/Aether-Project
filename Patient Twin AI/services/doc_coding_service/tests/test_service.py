"""DocCodingService — T3.1 DoD (docs/10; docs/04 §4):
a discharge-summary sample yields coded Conditions/Medications/Observations;
sub-threshold codes stay `proposed` (await confirmation).
"""

from __future__ import annotations

from uuid import uuid4

from schemas.document import ClinicalDocument, CodeStatus, DocumentType, EntityType
from services.doc_coding_service.coder import DictionaryCoder, TermCode
from services.doc_coding_service.ocr import PassthroughOcr
from services.doc_coding_service.service import DocCodingService

_SAMPLE = (
    "DISCHARGE SUMMARY\n"
    "Diagnosis: Type 2 diabetes mellitus.\n"
    "Medications: Metformin 500mg twice daily.\n"
    "Labs: HbA1c 7.2 %.\n"
)

# Labelled DEV coding map (not clinical truth) for the sample.
_MAPPINGS = {
    "type 2 diabetes": TermCode(
        EntityType.CONDITION, "SNOMED CT", "44054006", "Type 2 diabetes mellitus", 0.90
    ),
    "metformin": TermCode(EntityType.MEDICATION, "RxNorm", "6809", "Metformin", 0.80),
    "hba1c": TermCode(
        EntityType.OBSERVATION, "LOINC", "4548-4", "Hemoglobin A1c", 0.40, value="7.2", unit="%"
    ),
}
_THRESHOLDS = {"condition": 0.7, "medication": 0.7, "observation": 0.7}


def _service(thresholds: dict[str, float] | None = None) -> DocCodingService:
    return DocCodingService(
        ocr=PassthroughOcr(), coder=DictionaryCoder(_MAPPINGS), thresholds=thresholds
    )


def _doc() -> ClinicalDocument:
    return ClinicalDocument(
        patient_id=uuid4(), doc_type=DocumentType.DISCHARGE_SUMMARY, text=_SAMPLE
    )


def test_discharge_summary_yields_coded_entities() -> None:
    result = _service(_THRESHOLDS).ingest(_doc())
    by_type = {e.entity_type: e for e in result.entities}
    assert by_type[EntityType.CONDITION].code == "44054006"
    assert by_type[EntityType.MEDICATION].code == "6809"
    assert by_type[EntityType.OBSERVATION].code == "4548-4"
    assert result.coder_version == "dictionary-dev"


def test_confidence_gate_sets_committed_vs_proposed() -> None:
    result = _service(_THRESHOLDS).ingest(_doc())
    status = {e.entity_type: e.status for e in result.entities}
    assert status[EntityType.CONDITION] is CodeStatus.COMMITTED  # 0.90 >= 0.7
    assert status[EntityType.MEDICATION] is CodeStatus.COMMITTED  # 0.80 >= 0.7
    assert status[EntityType.OBSERVATION] is CodeStatus.PROPOSED  # 0.40 < 0.7 — awaits confirmation


def test_unset_thresholds_leave_everything_proposed() -> None:
    # Shipped stub state: no threshold => nothing auto-commits (docs/04 §4).
    result = _service(thresholds={}).ingest(_doc())
    assert all(e.status is CodeStatus.PROPOSED for e in result.entities)


def test_ocr_used_when_no_inline_text() -> None:
    # PassthroughOcr requires inline text; a uri-only doc with no OCR text raises.
    import pytest

    doc = ClinicalDocument(
        patient_id=uuid4(), doc_type=DocumentType.DISCHARGE_SUMMARY, uri="file:///x.pdf"
    )
    with pytest.raises(ValueError, match="inline document.text"):
        _service(_THRESHOLDS).ingest(doc)
