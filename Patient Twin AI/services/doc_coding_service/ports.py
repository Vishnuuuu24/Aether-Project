"""Service-local ports for the document-coding pipeline (docs/04 §4).

Not part of the canonical ai/interfaces set (docs/02 §6) — these are swappable
model backends internal to this service, like ConsentProvider / PSGStore elsewhere.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas.document import ClinicalDocument, CodedEntity, DocumentType


@runtime_checkable
class OcrEngine(Protocol):
    def extract_text(self, document: ClinicalDocument) -> str: ...


@runtime_checkable
class ClinicalCoder(Protocol):
    version: str

    def code(self, text: str, *, doc_type: DocumentType) -> list[CodedEntity]: ...
