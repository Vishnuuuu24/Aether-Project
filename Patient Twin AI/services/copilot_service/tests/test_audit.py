"""The copilot emits a hash-chained audit record for every output (docs/06 §8)."""

from __future__ import annotations

from core.audit import AuditWriter, InMemoryAuditStore, verify_chain
from schemas.audit import AuditAction
from schemas.consent import ConsentScope
from services.copilot_service.audit import AuditWriterSink
from services.copilot_service.orchestrator import Copilot
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import PolicyRuleSet

from ._helpers import NOW, PATIENT_ID, VERSIONS, grounded_proposal, kb_evidence, make_projection
from .test_orchestrator import FakeGateway, FakeRetriever


def test_output_is_audited_into_a_verifiable_chain() -> None:
    store = InMemoryAuditStore()
    copilot = Copilot(
        retriever=FakeRetriever(kb_evidence()),
        gateway=FakeGateway(grounded_proposal()),
        policy=PolicyEngine(PolicyRuleSet()),
        versions=VERSIONS,
        audit_sink=AuditWriterSink(AuditWriter(store)),
    )
    out = copilot.answer(
        patient_id=PATIENT_ID,
        projection=make_projection(),
        query="why is my HR up?",
        consented_scopes=[ConsentScope.COPILOT],
        now=NOW,
    )

    records = store.records
    assert len(records) == 1
    rec = records[0]
    assert rec.action == AuditAction.POLICY_DECISION
    assert rec.patient_id == PATIENT_ID
    assert str(out.output_id) in rec.output_refs
    assert rec.versions["model"] == VERSIONS.model
    verify_chain(records)  # hash chain intact
