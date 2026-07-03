"""Failure-mode defaults verification (docs/06 §9; T5.3 DoD).

One acceptance test per row of the docs/06 §9 table, exercised end-to-end through
the REAL Copilot orchestrator + Policy Engine + de-id gate. This is the concrete
"failure-mode defaults verified" artifact: if any default regresses, one of these
fails.

| Failure                     | Verified default behaviour                        |
|-----------------------------|---------------------------------------------------|
| LLM Gateway unavailable     | abstain; never ungrounded generation              |
| Retrieval empty             | abstain / info-only; "no supporting evidence"     |
| Baseline population-fallback| proceed but flag non-personalised; lower confidence|
| De-identification uncertain | block external egress (default-deny)              |
| Schema / grounding violation| suppress/abstain, logged (persisted + audited)    |
"""

from __future__ import annotations

import pytest

from ai.llm.client import LLMUnavailable
from ai.llm.deid import EgressBlocked, assert_clean_for_egress
from schemas.consent import ConsentScope
from schemas.output_contract import OutputContract, PolicyDecision, RecommendedAction
from schemas.psg import PSGProjection
from services.copilot_service.orchestrator import Copilot
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import PolicyRuleSet, RedFlagRule
from services.policy_engine.tests._fixtures import (
    NOW,
    PATIENT_ID,
    VERSIONS,
    grounded_proposal,
    kb_evidence,
    make_projection,
)

from .test_orchestrator import FakeGateway, FakeRetriever, RecordingSink


def _copilot(
    gateway: FakeGateway,
    *,
    evidence: list | None = None,
    ruleset: PolicyRuleSet | None = None,
    sinks: dict | None = None,
) -> Copilot:
    sinks = sinks or {}
    return Copilot(
        retriever=FakeRetriever(kb_evidence() if evidence is None else evidence),
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


def test_llm_gateway_unavailable_abstains_never_ungrounded() -> None:
    out = _answer(
        _copilot(FakeGateway(raises=LLMUnavailable("connection refused"))), make_projection()
    )
    assert out.policy.decision == PolicyDecision.ABSTAIN
    assert out.abstained.value is True
    assert out.evidence == []  # no fabricated grounding
    assert "R0_gateway_unavailable" in out.policy.rule_ids


def test_retrieval_empty_abstains_with_no_supporting_evidence() -> None:
    # Empty corpus → gateway has nothing to ground on → proposes without evidence.
    proposal = grounded_proposal(with_evidence=False)
    out = _answer(_copilot(FakeGateway(proposal), evidence=[]), make_projection())
    assert out.policy.decision == PolicyDecision.ABSTAIN
    assert "R2_grounding" in out.policy.rule_ids


def test_population_fallback_proceeds_but_downgrades_and_caps_confidence() -> None:
    out = _answer(
        _copilot(FakeGateway(grounded_proposal(population_fallback=True))),
        make_projection(population_fallback=True),
    )
    assert out.policy.decision == PolicyDecision.DOWNGRADED
    assert out.confidence <= 0.5
    assert "population" in out.message.lower()
    assert "R7_population_fallback" in out.policy.rule_ids


def test_deidentification_uncertain_blocks_egress() -> None:
    # Unit: the default-deny gate refuses an identifier-bearing payload.
    with pytest.raises(EgressBlocked):
        assert_clean_for_egress("contact patient at jane.doe@example.com")
    # End-to-end: a blocked-egress gateway failure abstains, never leaks.
    out = _answer(_copilot(FakeGateway(raises=EgressBlocked(["email"]))), make_projection())
    assert out.policy.decision == PolicyDecision.ABSTAIN


def test_grounding_violation_abstains_and_is_logged() -> None:
    store, audit = RecordingSink(), RecordingSink()
    # A proposal citing an unprovided ref is a grounding violation.
    out = _answer(
        _copilot(
            FakeGateway(grounded_proposal(invented_ref=True)),
            sinks={"store": store, "audit": audit},
        ),
        make_projection(),
    )
    assert out.policy.decision == PolicyDecision.ABSTAIN
    assert len(store.items) == 1  # logged: persisted
    assert len(audit.items) == 1  # logged: audited


def test_scope_violation_is_suppressed_and_logged() -> None:
    store, audit = RecordingSink(), RecordingSink()
    ruleset = PolicyRuleSet(prohibited_terms=("myocardial infarction",))
    out = _answer(
        _copilot(
            FakeGateway(grounded_proposal(message="This looks like a myocardial infarction.")),
            ruleset=ruleset,
            sinks={"store": store, "audit": audit},
        ),
        make_projection(),
    )
    assert out.policy.decision == PolicyDecision.SUPPRESSED
    assert len(store.items) == 1 and len(audit.items) == 1


def test_configured_red_flag_pattern_escalates_when_set() -> None:
    # Verifies the escalation default is data-driven: a configured acute pattern (here
    # supplied explicitly, standing in for clinician config) forces escalation.
    ruleset = PolicyRuleSet(
        red_flags=(
            RedFlagRule(
                id="RF_test",
                action=RecommendedAction.SEEK_CARE,
                any_active_event_type=("generic_event",),
            ),
        )
    )
    esc = RecordingSink()
    out = _answer(
        _copilot(FakeGateway(grounded_proposal()), ruleset=ruleset, sinks={"escalation": esc}),
        make_projection(high_severity_event=True, event_type="generic_event"),
    )
    assert out.escalation.triggered is True
    assert len(esc.items) == 1
