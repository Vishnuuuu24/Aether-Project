"""Copilot orchestration: the projection→retrieve→propose→policy→render path, and its
fail-safe / side-effect guarantees (docs/07 §5; docs/10 T4.2 DoD)."""

from __future__ import annotations

from ai.llm.client import LLMUnavailable
from ai.llm.deid import EgressBlocked
from schemas.consent import ConsentScope
from schemas.output_contract import OutputContract, PolicyDecision, ProposedOutput
from schemas.psg import PSGProjection
from schemas.retrieval import EvidenceChunk, RetrievalScope
from services.copilot_service.orchestrator import Copilot
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import PolicyRuleSet

# Reuse the policy engine's projection/proposal builders (DRY).
from services.policy_engine.tests._fixtures import (
    NOW,
    PATIENT_ID,
    VERSIONS,
    grounded_proposal,
    kb_evidence,
    make_projection,
)


class FakeRetriever:
    def __init__(self, evidence: list[EvidenceChunk]) -> None:
        self._evidence = evidence
        self.last_scope: RetrievalScope | None = None

    def search(self, query: str, scope: RetrievalScope, *, k: int = 10) -> list[EvidenceChunk]:
        self.last_scope = scope
        return self._evidence


class FakeGateway:
    def __init__(
        self, proposal: ProposedOutput | None = None, raises: Exception | None = None
    ) -> None:
        self._proposal = proposal
        self._raises = raises

    def propose(self, *, query, projection, evidence, locale="en") -> ProposedOutput:
        if self._raises is not None:
            raise self._raises
        assert self._proposal is not None
        return self._proposal


class RecordingSink:
    def __init__(self) -> None:
        self.items: list[OutputContract] = []

    def save(self, output: OutputContract) -> None:
        self.items.append(output)

    def enqueue(self, output: OutputContract) -> None:
        self.items.append(output)

    def record(self, output: OutputContract) -> None:
        self.items.append(output)


def _copilot(
    gateway: FakeGateway, *, ruleset: PolicyRuleSet | None = None, sinks: dict | None = None
) -> Copilot:
    sinks = sinks or {}
    return Copilot(
        retriever=FakeRetriever(kb_evidence()),
        gateway=gateway,
        policy=PolicyEngine(ruleset or PolicyRuleSet()),
        versions=VERSIONS,
        output_store=sinks.get("store"),
        escalation_sink=sinks.get("escalation"),
        audit_sink=sinks.get("audit"),
    )


def _answer(copilot: Copilot, projection: PSGProjection) -> OutputContract:
    return copilot.answer(
        patient_id=PATIENT_ID,
        projection=projection,
        query="why is my resting heart rate up?",
        consented_scopes=[ConsentScope.COPILOT, ConsentScope.VITALS],
        now=NOW,
    )


def test_happy_path_returns_approved_contract() -> None:
    store, audit, esc = RecordingSink(), RecordingSink(), RecordingSink()
    copilot = _copilot(
        FakeGateway(grounded_proposal()),
        sinks={"store": store, "audit": audit, "escalation": esc},
    )
    out = _answer(copilot, make_projection())
    assert isinstance(out, OutputContract)
    assert out.policy.decision == PolicyDecision.APPROVED
    assert out.evidence  # every approved claim carries evidence
    assert len(store.items) == 1  # persisted
    assert len(audit.items) == 1  # audited
    assert esc.items == []  # nothing to escalate


def test_returns_only_output_contract_never_raw_llm() -> None:
    copilot = _copilot(FakeGateway(grounded_proposal()))
    out = _answer(copilot, make_projection())
    assert not isinstance(out, ProposedOutput)
    assert isinstance(out, OutputContract)


def test_retrieval_is_consent_scoped() -> None:
    retriever = FakeRetriever(kb_evidence())
    copilot = Copilot(
        retriever=retriever,
        gateway=FakeGateway(grounded_proposal()),
        policy=PolicyEngine(PolicyRuleSet()),
        versions=VERSIONS,
    )
    copilot.answer(
        patient_id=PATIENT_ID,
        projection=make_projection(),
        query="q",
        consented_scopes=[ConsentScope.COPILOT],
        now=NOW,
    )
    assert retriever.last_scope is not None
    assert retriever.last_scope.patient_id == PATIENT_ID
    assert ConsentScope.COPILOT in retriever.last_scope.consented_scopes


def test_gateway_unavailable_abstains() -> None:
    store, audit = RecordingSink(), RecordingSink()
    copilot = _copilot(
        FakeGateway(raises=LLMUnavailable("connection refused")),
        sinks={"store": store, "audit": audit},
    )
    out = _answer(copilot, make_projection())
    assert out.policy.decision == PolicyDecision.ABSTAIN
    assert out.abstained.value is True
    assert "R0_gateway_unavailable" in out.policy.rule_ids
    assert len(store.items) == 1 and len(audit.items) == 1  # still persisted + audited


def test_blocked_egress_abstains() -> None:
    copilot = _copilot(FakeGateway(raises=EgressBlocked(["email"])))
    out = _answer(copilot, make_projection())
    assert out.policy.decision == PolicyDecision.ABSTAIN


def test_red_flag_escalation_is_enqueued() -> None:
    esc = RecordingSink()
    copilot = _copilot(FakeGateway(grounded_proposal()), sinks={"escalation": esc})
    out = _answer(copilot, make_projection(high_severity_event=True))
    assert out.escalation.triggered is True
    assert len(esc.items) == 1  # queued for clinician review


def test_escalation_fires_even_when_gateway_dead() -> None:
    # Acute safety must survive a dead gateway: red flag depends on the projection,
    # not on the LLM. (Gateway raises, but the HIGH-severity event still escalates.)
    esc = RecordingSink()
    copilot = _copilot(FakeGateway(raises=LLMUnavailable("down")), sinks={"escalation": esc})
    out = _answer(copilot, make_projection(high_severity_event=True))
    assert out.escalation.triggered is True
    assert len(esc.items) == 1
