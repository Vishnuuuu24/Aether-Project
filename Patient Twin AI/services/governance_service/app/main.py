"""governance-service API (docs/07 §2 & §7; T5.1).

Four capability groups, all backed by ONE hash-chained audit trail:

- Consent lifecycle: `POST/DELETE/GET /v1/patients/{id}/consent` (docs/07 §2).
- Audit query:       `GET  /v1/audit` — the tamper-evident trail, filtered and
                     re-verified before return (docs/07 §7).
- Outcome capture:   `POST /v1/outcomes` — outer-loop labels linked to prior
                     outputs and versions (docs/11 §3).
- Version registry:  `GET  /v1/versions` — active model/ruleset/prompt/etc.

Dev wiring is in-memory. Production injects a Postgres-backed audit store (one
transaction) and the same components via DI. Nothing here is writable by the LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from core.audit import AuditWriter, InMemoryAuditStore
from core.versioning import VersionRegistry
from schemas.audit import AuditAction, AuditRecord
from schemas.consent import Consent, ConsentScope
from schemas.outcome import Outcome, OutcomeSource, OutcomeType

from ..audit_query import query_audit
from ..consent import ConsentLedger
from ..outcomes import OutcomeStore

app = FastAPI(title="patient-copilot-governance-service", version="0.0.1")

# Dev wiring (in-memory). ONE audit store so consent changes and outcome captures
# share a single chain. Production swaps in the Postgres-backed store via DI.
_audit_store = InMemoryAuditStore()
_audit_writer = AuditWriter(_audit_store)
_consent_ledger = ConsentLedger(_audit_writer)
_outcome_store = OutcomeStore(_audit_writer)
_versions = VersionRegistry.from_env()


def get_consent_ledger() -> ConsentLedger:
    return _consent_ledger


def get_outcome_store() -> OutcomeStore:
    return _outcome_store


def get_audit_store() -> InMemoryAuditStore:
    return _audit_store


def get_versions() -> VersionRegistry:
    return _versions


class ConsentRequest(BaseModel):
    scope: list[ConsentScope] = Field(min_length=1)
    version: str = Field(min_length=1)


class OutcomeRequest(BaseModel):
    """Client-supplied outcome. `outcome_id` and `recorded_at` are server-stamped;
    the client asserts what happened, when, and which prior outputs it bears on."""

    patient_id: UUID
    outcome_type: OutcomeType
    occurred_at: datetime
    detail: str = Field(min_length=1)
    code: str | None = None
    linked_output_ids: list[UUID] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    source: OutcomeSource


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz() -> dict[str, bool]:
    return {"ready": True}


@app.post("/v1/patients/{patient_id}/consent")
async def grant_consent(
    patient_id: UUID,
    body: ConsentRequest,
    ledger: Annotated[ConsentLedger, Depends(get_consent_ledger)],
) -> Consent:
    return ledger.grant(patient_id, scope=body.scope, version=body.version)


@app.get("/v1/patients/{patient_id}/consent")
async def get_consent(
    patient_id: UUID,
    ledger: Annotated[ConsentLedger, Depends(get_consent_ledger)],
) -> Consent:
    consent = ledger.get_consent(patient_id)
    if consent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no consent record for patient"
        )
    return consent


@app.delete("/v1/patients/{patient_id}/consent")
async def revoke_consent(
    patient_id: UUID,
    ledger: Annotated[ConsentLedger, Depends(get_consent_ledger)],
) -> Consent:
    consent = ledger.revoke(patient_id)
    if consent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no consent record for patient"
        )
    return consent


@app.get("/v1/audit")
async def get_audit(
    store: Annotated[InMemoryAuditStore, Depends(get_audit_store)],
    patient_id: UUID | None = None,
    action: AuditAction | None = None,
    since: datetime | None = None,
    output_id: UUID | None = None,
) -> list[AuditRecord]:
    return query_audit(
        store, patient_id=patient_id, action=action, since=since, output_id=output_id
    )


@app.get("/v1/versions")
async def get_versions_endpoint(
    versions: Annotated[VersionRegistry, Depends(get_versions)],
) -> dict[str, str]:
    return versions.current().as_dict()


@app.post("/v1/outcomes", status_code=status.HTTP_201_CREATED)
async def record_outcome(
    body: OutcomeRequest,
    store: Annotated[OutcomeStore, Depends(get_outcome_store)],
) -> Outcome:
    outcome = Outcome(
        patient_id=body.patient_id,
        outcome_type=body.outcome_type,
        occurred_at=body.occurred_at,
        detail=body.detail,
        code=body.code,
        linked_output_ids=body.linked_output_ids,
        versions=body.versions,
        source=body.source,
        recorded_at=datetime.now(UTC),
    )
    return store.record(outcome)
