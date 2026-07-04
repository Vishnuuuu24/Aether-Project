"""ingestion-service API (docs/07 §3).

Routes, all funnelling accepted rows through the ONE normaliser + consent gate:

  * `POST /v1/ingest/readings` — batch of `Reading[]`-shaped JSON. Per-item outcomes:
    200 when all accepted, 207 when any are rejected (validated individually so one
    bad item doesn't 422 the batch).
  * `POST /v1/ingest/adapters/{adapter}/webhook` — a device push (HealthKit / Health
    Connect / Fitbit); mapped by the named adapter, then normalised.
  * `POST /v1/ingest/replay` — dev-only dataset replay harness (env-gated off in prod).

Consent is deny-by-default (docs/02 §2): the dev wiring uses an empty in-memory
consent store, so readings are rejected until the patient's `vitals` consent is
seeded (the governance service backs this in production).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import Body, Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from core.audit import AuditWriter, InMemoryAuditStore
from schemas.reading import IngestBatchResult

from ..adapters import fitbit, health_connect, healthkit
from ..adapters.replay import stream_dataset
from ..consent import StaticConsentProvider
from ..service import IngestionService
from ..sink import InMemoryReadingSink

# Adapter dispatch: name → its raw-payload → canonical-reading mapper (docs/07 §3).
_ADAPTERS: dict[str, Callable[..., Iterator[dict[str, Any]]]] = {
    healthkit.ADAPTER_NAME: healthkit.to_canonical,
    health_connect.ADAPTER_NAME: health_connect.to_canonical,
    fitbit.ADAPTER_NAME: fitbit.to_canonical,
}

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


class WebhookPayload(BaseModel):
    """A device push: the target patient plus the adapter's raw sample rows."""

    patient_id: UUID
    items: list[dict[str, Any]] = Field(default_factory=list)


class ReplayRequest(BaseModel):
    """Dev-only dataset replay (docs/07 §3). `speed` is advisory — the synchronous
    endpoint normalises the batch and reports counts; true timed replay is the CLI.
    """

    dataset: str = Field(min_length=1)
    patient_id: UUID
    speed: float = Field(default=1.0, gt=0.0)


def _replay_enabled() -> bool:
    # Dev-only (docs/15 T6.4). Production compose sets INGEST_ENABLE_REPLAY=0.
    return os.environ.get("INGEST_ENABLE_REPLAY", "1").lower() not in ("0", "false", "")


def _datasets_dir() -> Path:
    return Path(os.environ.get("INGEST_DATASETS_DIR", "datasets"))


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


@app.post("/v1/ingest/adapters/{adapter}/webhook")
async def ingest_webhook(
    adapter: str,
    body: WebhookPayload,
    response: Response,
    service: Annotated[IngestionService, Depends(get_service)],
) -> IngestBatchResult:
    mapper = _ADAPTERS.get(adapter)
    if mapper is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown adapter {adapter!r}; supported: {sorted(_ADAPTERS)}",
        )
    canonical = list(mapper(body.items, patient_id=str(body.patient_id)))
    result = service.ingest(canonical, adapter=adapter)
    response.status_code = status.HTTP_207_MULTI_STATUS if result.rejected else status.HTTP_200_OK
    return result


@app.post("/v1/ingest/replay")
async def ingest_replay(
    body: ReplayRequest,
    response: Response,
    service: Annotated[IngestionService, Depends(get_service)],
) -> IngestBatchResult:
    if not _replay_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="dataset replay is disabled"
        )
    path = _datasets_dir() / body.dataset / "S1.pkl"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"dataset file not found for {body.dataset!r} (see datasets/ README)",
        )
    try:
        stream = stream_dataset(body.dataset, path=path, patient_id=body.patient_id)
        result = service.ingest(stream, adapter="replay")
    except ValueError as exc:  # unknown dataset name
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    response.status_code = status.HTTP_207_MULTI_STATUS if result.rejected else status.HTTP_200_OK
    return result
