"""patient-state-engine API (docs/07; T1.4).

`GET /v1/state/{patient_id}` — the consent-scoped PSG projection. It carries no raw
signals and no reading-level data (CLAUDE.md principle 2). Deny-by-default: a patient
with no consent scope in force gets 403; an unknown patient gets 404.

Dev wiring is in-memory. Production injects a Postgres-backed `SqlAlchemyPSGStore`
+ `SqlAlchemyAuditStore` (sharing one transaction) and a governance-backed consent
and profile provider.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status

from core.audit import AuditWriter, InMemoryAuditStore
from core.auth.errors import ConsentError
from core.versioning import VersionRegistry
from schemas.psg import PSGProjection

from ..consent import StaticConsentProvider
from ..profile import StaticProfileProvider
from ..service import PatientStateEngine, ProfileNotFoundError
from ..store import InMemoryPSGStore

app = FastAPI(title="patient-copilot-state-engine", version="0.0.1")

# Dev wiring (in-memory). Production swaps the store/audit/providers via DI.
_consent_provider = StaticConsentProvider()
_profile_provider = StaticProfileProvider()
_engine = PatientStateEngine(
    store=InMemoryPSGStore(),
    consent_provider=_consent_provider,
    audit_writer=AuditWriter(InMemoryAuditStore()),
    versions=VersionRegistry.from_env().current(),
    profile_provider=_profile_provider,
)


def get_engine() -> PatientStateEngine:
    return _engine


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz() -> dict[str, bool]:
    return {"ready": True}


@app.get("/v1/state/{patient_id}")
async def get_state(
    patient_id: UUID, engine: Annotated[PatientStateEngine, Depends(get_engine)]
) -> PSGProjection:
    try:
        return engine.build_projection(patient_id)
    except ProfileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient not found"
        ) from exc
    except ConsentError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
