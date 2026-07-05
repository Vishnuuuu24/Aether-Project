"""doc-coding-service API (docs/07 §3; docs/15 T6.4).

`POST /v1/ingest/documents` — multipart upload (file + `doc_type` + `patient_id`)
→ OCR → clinical coding → coded entities with a confidence gate. Sub-threshold
codes stay `proposed`; nothing is auto-committed without a configured threshold
(fail-safe, docs/04 §4). Committing the coded entities onto the PSG is the Patient
State Engine's job — this service only produces the coding result.

v1 handles **text** documents (OCR of scanned binaries is the deferred `DoclingOcr`
path). The dev wiring ships a coder with NO clinical mappings, so it fabricates no
codes (CLAUDE.md); production injects `MedCatCoder` + a licensed model pack via DI.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from core.observability import install_observability
from schemas.document import ClinicalDocument, DocumentCodingResult, DocumentType

from ..coder import DictionaryCoder
from ..ocr import PassthroughOcr
from ..service import DocCodingService

app = FastAPI(title="patient-copilot-doc-coding-service", version="0.0.1")
install_observability(app, service="doc-coding")

# Dev wiring: passthrough OCR + an EMPTY dictionary coder (emits no codes without a
# clinical map) + UNSET thresholds (nothing auto-commits). Production swaps in the
# real OCR/coder and thresholds via DI.
_service = DocCodingService(ocr=PassthroughOcr(), coder=DictionaryCoder({}), thresholds={})


def get_service() -> DocCodingService:
    return _service


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz() -> dict[str, bool]:
    return {"ready": True}


@app.post("/v1/ingest/documents", status_code=status.HTTP_201_CREATED)
async def ingest_document(
    service: Annotated[DocCodingService, Depends(get_service)],
    patient_id: Annotated[UUID, Form()],
    doc_type: Annotated[DocumentType, Form()],
    file: Annotated[UploadFile, File()],
) -> DocumentCodingResult:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="document must be UTF-8 text; binary OCR is deferred (v1 is text-only)",
        ) from exc

    try:
        document = ClinicalDocument(patient_id=patient_id, doc_type=doc_type, text=text)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="document has no readable text",
        ) from exc

    try:
        return service.ingest(document)
    except ValueError as exc:  # e.g. blank text and no OCR source
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
