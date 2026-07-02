"""DocCodingService — OCR + code + confidence gate (docs/04 §4; T3.1).

Runs the pipeline and applies the confidence gate: an entity is `committed` only
when its confidence meets the configured per-type threshold; otherwise it stays
`proposed` (fail-safe with no threshold set — nothing is silently committed).

This service produces coded entities; committing them onto the PSG (append-only +
audited, DOCUMENTS-scoped) is the Patient State Engine's job (docs/02 §4).
"""

from __future__ import annotations

from schemas.document import (
    ClinicalDocument,
    CodedEntity,
    CodeStatus,
    DocumentCodingResult,
)

from .ports import ClinicalCoder, OcrEngine


class DocCodingService:
    def __init__(
        self,
        *,
        ocr: OcrEngine,
        coder: ClinicalCoder,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._ocr = ocr
        self._coder = coder
        self._thresholds = thresholds or {}

    def ingest(self, document: ClinicalDocument) -> DocumentCodingResult:
        if document.text and document.text.strip():
            text = document.text
        else:
            text = self._ocr.extract_text(document)
        entities = [self._gate(e) for e in self._coder.code(text, doc_type=document.doc_type)]
        return DocumentCodingResult(
            document_id=document.document_id,
            patient_id=document.patient_id,
            doc_type=document.doc_type,
            entities=entities,
            coder_version=self._coder.version,
        )

    def _gate(self, entity: CodedEntity) -> CodedEntity:
        threshold = self._thresholds.get(entity.entity_type.value)
        committed = threshold is not None and entity.confidence >= threshold
        status = CodeStatus.COMMITTED if committed else CodeStatus.PROPOSED
        return entity.model_copy(update={"status": status})
