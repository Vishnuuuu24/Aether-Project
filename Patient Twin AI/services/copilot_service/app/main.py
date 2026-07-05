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

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ai.llm.gateway import Gateway
from ai.retrieval import (
    HashEmbedder,
    HybridRetriever,
    InMemoryVectorStore,
    LexicalReranker,
    QdrantVectorStore,
)
from ai.retrieval.ports import VectorStore
from core.audit import AuditWriter, InMemoryAuditStore
from core.audit.sql_store import SqlAlchemyAuditStore
from core.auth.errors import ConsentError
from core.db import persistence_backend, request_session
from core.observability import install_observability
from core.versioning import VersionRegistry
from schemas.consent import ConsentScope
from schemas.output_contract import OutputContract, PolicyDecision
from schemas.psg import PSGProjection
from services.patient_state_engine.consent import StaticConsentProvider
from services.patient_state_engine.profile import StaticProfileProvider
from services.patient_state_engine.service import PatientStateEngine, ProfileNotFoundError
from services.patient_state_engine.store import InMemoryPSGStore
from services.patient_state_engine.wiring import build_sql_engine
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import load_policy_rules

from ..audit import AuditWriterSink
from ..escalation import (
    EscalationNotFoundError,
    EscalationQueue,
    EscalationRecord,
    EscalationStatus,
)
from ..orchestrator import Copilot
from ..output_store import SqlOutputStore

app = FastAPI(title="patient-copilot-copilot-service", version="0.0.1")
install_observability(app, service="copilot")

_versions = VersionRegistry.from_env().current()
_audit_writer = AuditWriter(InMemoryAuditStore())
# The copilot's escalation sink and the clinician read/ack surface are the SAME queue.
_escalation_queue = EscalationQueue(_audit_writer)
_state_engine = PatientStateEngine(
    store=InMemoryPSGStore(),
    consent_provider=StaticConsentProvider(),
    audit_writer=AuditWriter(InMemoryAuditStore()),
    versions=_versions,
    profile_provider=StaticProfileProvider(),
)


# Empty-KB dev retriever (real HybridRetriever with deterministic dev adapters). The
# retriever/gateway/policy are stateless and shared; only the audit sink (and the
# projection source) become per-request + DB-backed in `postgres` mode.
def _build_vector_store() -> VectorStore:
    # Persist vectors to Qdrant in production posture; the in-memory cosine store
    # drives dev + the fast suite. (No connection happens here: the empty dev KB is
    # never upserted, and QdrantVectorStore connects lazily.)
    url = os.environ.get("QDRANT_URL")
    if persistence_backend() == "postgres" and url:
        return QdrantVectorStore(url=url)
    return InMemoryVectorStore()


_retriever = HybridRetriever(
    corpus=[],
    embedder=HashEmbedder(),
    reranker=LexicalReranker(),
    vector_store=_build_vector_store(),
)
_gateway = Gateway.from_config()
_policy = PolicyEngine(load_policy_rules())
_copilot = Copilot(
    retriever=_retriever,
    gateway=_gateway,
    policy=_policy,
    versions=_versions,
    audit_sink=AuditWriterSink(_audit_writer),
    escalation_sink=_escalation_queue,
)

# FastAPI caches this per request → the projection read and the output audit write
# share ONE transaction / one chain append when DB-backed.
_Session = Annotated[Session | None, Depends(request_session)]


def get_state_engine(session: _Session) -> PatientStateEngine:
    if session is None:
        return _state_engine
    return build_sql_engine(session, _versions)


def get_copilot(session: _Session) -> Copilot:
    if session is None:
        return _copilot
    return Copilot(
        retriever=_retriever,
        gateway=_gateway,
        policy=_policy,
        versions=_versions,
        output_store=SqlOutputStore(session),
        audit_sink=AuditWriterSink(AuditWriter(SqlAlchemyAuditStore(session))),
        escalation_sink=_escalation_queue,
    )


def get_escalation_queue() -> EscalationQueue:
    return _escalation_queue


class CopilotQuery(BaseModel):
    query: str = Field(min_length=1)
    locale: str = "en"


class AckRequest(BaseModel):
    clinician: str = Field(min_length=1)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz() -> dict[str, bool]:
    return {"ready": True}


def _answer(
    patient_id: UUID, body: CopilotQuery, engine: PatientStateEngine, copilot: Copilot
) -> OutputContract:
    """Shared path for both the JSON and SSE variants: build the consent-scoped
    projection, then run the full retrieve→propose→Policy pipeline. The returned
    contract is ALWAYS Policy-decided — nothing downstream sees raw LLM output.
    """
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


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_stream(output: OutputContract) -> Iterator[str]:
    """Emit the Policy-vetted output as SSE. Message tokens are streamed ONLY for a
    real answer (approved / downgraded); abstained or suppressed decisions stream no
    token content — the client still receives the full structured contract in the
    terminal `result` event. Nothing here is raw LLM output: `output` is post-Policy.
    """
    streamable = (
        output.policy.decision in (PolicyDecision.APPROVED, PolicyDecision.DOWNGRADED)
        and not output.abstained.value
    )
    if streamable:
        for token in output.message.split():
            yield _sse("token", {"text": token})
    yield _sse("result", output.model_dump(mode="json"))
    yield _sse("done", {})


@app.post("/v1/patients/{patient_id}/copilot/query")
async def copilot_query(
    patient_id: UUID,
    body: CopilotQuery,
    engine: Annotated[PatientStateEngine, Depends(get_state_engine)],
    copilot: Annotated[Copilot, Depends(get_copilot)],
) -> OutputContract:
    return _answer(patient_id, body, engine, copilot)


@app.post("/v1/patients/{patient_id}/copilot/query:stream")
async def copilot_query_stream(
    patient_id: UUID,
    body: CopilotQuery,
    engine: Annotated[PatientStateEngine, Depends(get_state_engine)],
    copilot: Annotated[Copilot, Depends(get_copilot)],
) -> StreamingResponse:
    # The full Policy decision runs HERE, before any bytes stream — so the SSE body
    # can never carry un-vetted content (docs/07 §5).
    output = _answer(patient_id, body, engine, copilot)
    return StreamingResponse(_sse_stream(output), media_type="text/event-stream")


@app.get("/v1/escalations")
async def list_escalations(
    queue: Annotated[EscalationQueue, Depends(get_escalation_queue)],
    status: EscalationStatus = EscalationStatus.OPEN,
) -> list[EscalationRecord]:
    """Clinician queue of red-flag / high-severity outputs (docs/07 §6). Read-only:
    listing never alters queue state.
    """
    return queue.list(status=status)


@app.post("/v1/escalations/{output_id}/ack")
async def ack_escalation(
    output_id: UUID,
    body: AckRequest,
    queue: Annotated[EscalationQueue, Depends(get_escalation_queue)],
) -> EscalationRecord:
    """Record + audit a clinician acknowledgement. Does not mutate or re-open the
    underlying output (docs/15 T6.3).
    """
    try:
        return queue.acknowledge(output_id, clinician=body.clinician, now=datetime.now(UTC))
    except EscalationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="no escalation for output"
        ) from exc
