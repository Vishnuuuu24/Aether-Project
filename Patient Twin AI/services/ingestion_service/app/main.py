"""ingestion-service API (docs/07 §3).

`POST /v1/ingest/readings` — batch ingest of `Reading[]`. Per-item outcomes: 200
when all accepted, 207 (multi-status) when any are rejected. The raw body is a
list of JSON objects (NOT `list[Reading]`) so one bad item doesn't 422 the whole
batch — each item is validated individually and rejected with field errors.

Consent is deny-by-default (docs/02 §2): the dev wiring uses an empty in-memory
consent store, so readings are rejected until the patient's `vitals` consent is
seeded (the governance service backs this in production).
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Body, Depends, FastAPI, Response, status

from core.audit import AuditWriter, InMemoryAuditStore
from schemas.reading import IngestBatchResult

from ..consent import StaticConsentProvider
from ..service import IngestionService
from ..sink import InMemoryReadingSink

app = FastAPI(title="patient-copilot-ingestion-service", version="0.0.1")

# Dev wiring (in-memory). Production injects a Redis sink, a governance-backed
# consent provider, and a Postgres-backed audit store.
_consent_provider = StaticConsentProvider()
_service = IngestionService(
    consent_provider=_consent_provider,
    sink=InMemoryReadingSink(),
    audit_writer=AuditWriter(InMemoryAuditStore()),
)


def get_service() -> IngestionService:
    return _service


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz() -> dict[str, bool]:
    return {"ready": True}


@app.post("/v1/ingest/readings")
async def ingest_readings(
    response: Response,
    items: Annotated[list[dict[str, Any]], Body(...)],
    service: Annotated[IngestionService, Depends(get_service)],
) -> IngestBatchResult:
    result = service.ingest(items, adapter="readings")
    response.status_code = status.HTTP_207_MULTI_STATUS if result.rejected else status.HTTP_200_OK
    return result
