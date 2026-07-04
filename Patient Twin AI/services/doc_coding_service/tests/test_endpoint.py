"""POST /v1/ingest/documents — multipart → OCR → coding → confidence gate
(docs/07 §3; docs/15 T6.4). Sub-threshold codes stay `proposed`; nothing is
auto-committed without a configured threshold.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from httpx import Response

from schemas.document import EntityType
from services.doc_coding_service.app.main import app, get_service
from services.doc_coding_service.coder import DictionaryCoder, TermCode
from services.doc_coding_service.ocr import PassthroughOcr
from services.doc_coding_service.service import DocCodingService

_SAMPLE = (
    "DISCHARGE SUMMARY\n"
    "Diagnosis: Type 2 diabetes mellitus.\n"
    "Medications: Metformin 500mg twice daily.\n"
    "Labs: HbA1c 7.2 %.\n"
)
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


def _service(thresholds: dict[str, float]) -> DocCodingService:
    return DocCodingService(
        ocr=PassthroughOcr(), coder=DictionaryCoder(_MAPPINGS), thresholds=thresholds
    )


def _client(service: DocCodingService) -> TestClient:
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def _post(client: TestClient, body: bytes, *, doc_type: str = "discharge_summary") -> Response:
    return client.post(
        "/v1/ingest/documents",
        files={"file": ("doc.txt", body, "text/plain")},
        data={"patient_id": str(uuid4()), "doc_type": doc_type},
    )


def test_document_codes_entities_and_applies_confidence_gate() -> None:
    try:
        resp = _post(_client(_service(_THRESHOLDS)), _SAMPLE.encode())
        assert resp.status_code == 201
        entities = {e["entity_type"]: e for e in resp.json()["entities"]}
        assert entities["condition"]["code"] == "44054006"
        assert entities["condition"]["status"] == "committed"  # 0.90 >= 0.7
        assert entities["medication"]["status"] == "committed"  # 0.80 >= 0.7
        assert entities["observation"]["status"] == "proposed"  # 0.40 < 0.7 — awaits confirmation
    finally:
        app.dependency_overrides.clear()


def test_unset_thresholds_leave_everything_proposed() -> None:
    try:
        resp = _post(_client(_service({})), _SAMPLE.encode())
        assert resp.status_code == 201
        assert all(e["status"] == "proposed" for e in resp.json()["entities"])
    finally:
        app.dependency_overrides.clear()


def test_binary_upload_is_rejected_422() -> None:
    try:
        resp = _post(_client(_service(_THRESHOLDS)), b"\xff\xfe\x00binary")
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_blank_document_is_rejected_422() -> None:
    try:
        resp = _post(_client(_service(_THRESHOLDS)), b"   \n  ")
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()
