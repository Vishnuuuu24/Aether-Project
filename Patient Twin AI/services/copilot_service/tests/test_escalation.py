"""Clinician escalation queue + endpoints (docs/07 §6; docs/15 T6.3).

Covers the whole path: the copilot enqueues a red-flag output → the clinician lists
the open queue → acknowledges one (audited, actor=clinician, output untouched).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from core.audit import AuditWriter, InMemoryAuditStore, verify_chain
from schemas.audit import AuditAction, AuditActor
from schemas.consent import ConsentScope
from schemas.output_contract import OutputContract
from schemas.retrieval import EvidenceChunk, RetrievalScope
from services.copilot_service.app.main import app, get_escalation_queue
from services.copilot_service.escalation import (
    EscalationNotFoundError,
    EscalationQueue,
    EscalationStatus,
)
from services.copilot_service.orchestrator import Copilot
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import PolicyRuleSet
from services.policy_engine.tests._fixtures import (
    NOW,
    PATIENT_ID,
    VERSIONS,
    grounded_proposal,
    kb_evidence,
    make_projection,
)

ACK_TIME = datetime(2026, 7, 3, 15, 0, tzinfo=UTC)


def _queue() -> tuple[EscalationQueue, InMemoryAuditStore]:
    store = InMemoryAuditStore()
    return EscalationQueue(AuditWriter(store)), store


def _escalated_output() -> OutputContract:
    """A real forced-escalation output from the Policy Engine (HIGH-severity event)."""
    out = PolicyEngine(PolicyRuleSet()).decide(
        grounded_proposal(),
        make_projection(high_severity_event=True),
        patient_id=PATIENT_ID,
        evidence=kb_evidence(),
        versions=VERSIONS,
        now=NOW,
    )
    assert out.escalation.triggered is True
    return out


# -- queue unit behaviour ---------------------------------------------------------


def test_enqueue_then_list_open_and_acknowledge_audits_as_clinician() -> None:
    queue, audit = _queue()
    out = _escalated_output()
    queue.enqueue(out)

    open_records = queue.list(status=EscalationStatus.OPEN)
    assert len(open_records) == 1
    assert open_records[0].output.output_id == out.output_id

    acked = queue.acknowledge(out.output_id, clinician="dr-smith", now=ACK_TIME)
    assert acked.status is EscalationStatus.ACKNOWLEDGED
    assert acked.acknowledged_by == "dr-smith"
    assert acked.acknowledged_at == ACK_TIME
    # The underlying output is untouched (not mutated / re-opened).
    assert acked.output == out
    assert queue.list(status=EscalationStatus.OPEN) == []

    records = audit.records
    verify_chain(records)
    assert len(records) == 1
    assert records[0].actor is AuditActor.CLINICIAN
    assert records[0].action is AuditAction.ESCALATION_ACK
    assert records[0].input_refs == [f"output:{out.output_id}"]
    assert records[0].versions  # version-stamped from the output


def test_enqueue_is_idempotent_on_output_id() -> None:
    queue, _ = _queue()
    out = _escalated_output()
    queue.enqueue(out)
    queue.enqueue(out)
    assert len(queue.list(status=None)) == 1


def test_acknowledge_is_idempotent_and_writes_one_audit_record() -> None:
    queue, audit = _queue()
    out = _escalated_output()
    queue.enqueue(out)
    queue.acknowledge(out.output_id, clinician="a", now=ACK_TIME)
    second = queue.acknowledge(out.output_id, clinician="b", now=ACK_TIME)
    # Re-ack is a no-op: keeps the first ack, writes no second audit record.
    assert second.acknowledged_by == "a"
    assert len(audit.records) == 1


def test_acknowledge_unknown_raises() -> None:
    queue, _ = _queue()
    try:
        queue.acknowledge(uuid4(), clinician="x", now=ACK_TIME)
    except EscalationNotFoundError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("expected EscalationNotFoundError")


# -- copilot enqueue path ---------------------------------------------------------


class _FakeRetriever:
    def __init__(self, evidence: list[EvidenceChunk]) -> None:
        self._evidence = evidence

    def search(self, query: str, scope: RetrievalScope, *, k: int = 10) -> list[EvidenceChunk]:
        return self._evidence


class _FakeGateway:
    def __init__(self, proposal: object) -> None:
        self._proposal = proposal

    def propose(self, *, query, projection, evidence, locale="en"):
        return self._proposal


def test_copilot_enqueues_red_flag_into_the_queue() -> None:
    queue, _ = _queue()
    copilot = Copilot(
        retriever=_FakeRetriever(kb_evidence()),
        gateway=_FakeGateway(grounded_proposal()),
        policy=PolicyEngine(PolicyRuleSet()),
        versions=VERSIONS,
        escalation_sink=queue,
    )
    out = copilot.answer(
        patient_id=PATIENT_ID,
        projection=make_projection(high_severity_event=True),
        query="why is my heart rate high?",
        consented_scopes=[ConsentScope.COPILOT, ConsentScope.VITALS],
        now=NOW,
    )
    assert out.escalation.triggered is True
    queued = queue.list(status=EscalationStatus.OPEN)
    assert len(queued) == 1
    assert queued[0].output.output_id == out.output_id


# -- HTTP surface -----------------------------------------------------------------


def test_escalation_endpoints_list_ack_and_404() -> None:
    queue, _ = _queue()
    out = _escalated_output()
    queue.enqueue(out)

    app.dependency_overrides[get_escalation_queue] = lambda: queue
    try:
        client = TestClient(app)

        listed = client.get("/v1/escalations")
        assert listed.status_code == 200
        assert len(listed.json()) == 1
        assert listed.json()[0]["status"] == "open"

        ack = client.post(f"/v1/escalations/{out.output_id}/ack", json={"clinician": "dr-smith"})
        assert ack.status_code == 200
        body = ack.json()
        assert body["status"] == "acknowledged"
        assert body["acknowledged_by"] == "dr-smith"

        assert client.get("/v1/escalations").json() == []  # open queue drained
        assert len(client.get("/v1/escalations?status=acknowledged").json()) == 1

        missing = client.post(f"/v1/escalations/{uuid4()}/ack", json={"clinician": "x"})
        assert missing.status_code == 404
    finally:
        app.dependency_overrides.clear()
