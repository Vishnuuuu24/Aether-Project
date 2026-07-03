"""copilot-service API (docs/07 §5; T4.2).

`POST /v1/patients/{patient_id}/copilot/query` — the only route to a user-facing
answer. It assembles the consent-scoped PSG projection, retrieves evidence, has the
LLM Gateway propose, and lets the deterministic Policy Engine decide. It ALWAYS
returns a valid, Policy-approved `OutputContract` (including abstained / suppressed /
escalated cases) and never raw LLM output (docs/07 §10).

Deny-by-default: a patient with no consent scope in force gets 403; unknown → 404.

Dev wiring is in-memory with an empty KB and the real local Gateway — so with no model
served, the endpoint safely abstains (docs/06 §9). Production injects the real
State Engine, retriever corpus, gateway backend, and DB-backed stores via DI.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from ai.llm.gateway import Gateway
from ai.retrieval import HashEmbedder, HybridRetriever, InMemoryVectorStore, LexicalReranker
from core.audit import AuditWriter, InMemoryAuditStore
from core.auth.errors import ConsentError
from core.versioning import VersionRegistry
from schemas.consent import ConsentScope
from schemas.output_contract import OutputContract
from schemas.psg import PSGProjection
from services.patient_state_engine.consent import StaticConsentProvider
from services.patient_state_engine.profile import StaticProfileProvider
from services.patient_state_engine.service import PatientStateEngine, ProfileNotFoundError
from services.patient_state_engine.store import InMemoryPSGStore
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import load_policy_rules

from ..audit import AuditWriterSink
from ..orchestrator import Copilot

app = FastAPI(title="patient-copilot-copilot-service", version="0.0.1")

_versions = VersionRegistry.from_env().current()
_audit_writer = AuditWriter(InMemoryAuditStore())
_state_engine = PatientStateEngine(
    store=InMemoryPSGStore(),
    consent_provider=StaticConsentProvider(),
    audit_writer=AuditWriter(InMemoryAuditStore()),
    versions=_versions,
    profile_provider=StaticProfileProvider(),
)
# Empty-KB dev retriever (real HybridRetriever with deterministic dev adapters).
_retriever = HybridRetriever(
    corpus=[],
    embedder=HashEmbedder(),
    reranker=LexicalReranker(),
    vector_store=InMemoryVectorStore(),
)
_copilot = Copilot(
    retriever=_retriever,
    gateway=Gateway.from_config(),
    policy=PolicyEngine(load_policy_rules()),
    versions=_versions,
    audit_sink=AuditWriterSink(_audit_writer),
)


def get_state_engine() -> PatientStateEngine:
    return _state_engine


def get_copilot() -> Copilot:
    return _copilot


class CopilotQuery(BaseModel):
    query: str = Field(min_length=1)
    locale: str = "en"


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz() -> dict[str, bool]:
    return {"ready": True}


@app.post("/v1/patients/{patient_id}/copilot/query")
async def copilot_query(
    patient_id: UUID,
    body: CopilotQuery,
    engine: Annotated[PatientStateEngine, Depends(get_state_engine)],
    copilot: Annotated[Copilot, Depends(get_copilot)],
) -> OutputContract:
    try:
        projection: PSGProjection = engine.build_projection(patient_id)
    except ProfileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient not found"
        ) from exc
    except ConsentError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    valid_scopes = {c.value for c in ConsentScope}
    scopes = [ConsentScope(s) for s in projection.consent_scope if s in valid_scopes]
    return copilot.answer(
        patient_id=patient_id,
        projection=projection,
        query=body.query,
        consented_scopes=scopes,
        now=datetime.now(UTC),
        locale=body.locale,
    )
