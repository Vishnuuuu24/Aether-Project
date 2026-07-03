"""Policy Engine: ordered deterministic checks, escalation, abstention, mandatory
elements (docs/06; docs/10 T4.3 DoD)."""

from __future__ import annotations

from schemas.output_contract import (
    OutputType,
    PolicyDecision,
    RecommendedAction,
    Severity,
)
from schemas.output_contract import OutputType as OT
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import PolicyRuleSet, RedFlagRule

from ._fixtures import (
    NOW,
    PATIENT_ID,
    VERSIONS,
    grounded_proposal,
    kb_evidence,
    make_projection,
)


def _engine(ruleset: PolicyRuleSet | None = None) -> PolicyEngine:
    return PolicyEngine(ruleset or PolicyRuleSet())


def _decide(engine: PolicyEngine, proposal, projection):
    return engine.decide(
        proposal,
        projection,
        patient_id=PATIENT_ID,
        evidence=kb_evidence(),
        versions=VERSIONS,
        now=NOW,
    )


def test_grounded_proposal_is_approved_and_stamped() -> None:
    out = _decide(_engine(), grounded_proposal(), make_projection())
    assert out.policy.decision == PolicyDecision.APPROVED
    assert out.evidence  # claim carries evidence
    assert out.versions.model == "m1"
    assert out.disclaimer  # mandatory disclaimer present
    assert out.abstained.value is False


def test_every_output_carries_a_policy_record() -> None:
    # Approve, abstain, suppress, escalate — all must have a decision record.
    outs = [
        _decide(_engine(), grounded_proposal(), make_projection()),
        _decide(_engine(), grounded_proposal(with_evidence=False), make_projection()),
        _decide(
            _engine(PolicyRuleSet(prohibited_terms=("diagnos",))),
            grounded_proposal(message="This is a diagnosis of X."),
            make_projection(),
        ),
        _decide(_engine(), grounded_proposal(), make_projection(high_severity_event=True)),
    ]
    for out in outs:
        assert out.policy is not None
        assert out.policy.decision in set(PolicyDecision)
        assert out.policy.rule_ids


def test_ungrounded_proposal_abstains() -> None:
    out = _decide(_engine(), grounded_proposal(with_evidence=False), make_projection())
    assert out.policy.decision == PolicyDecision.ABSTAIN
    assert out.abstained.value is True


def test_invented_citation_abstains() -> None:
    out = _decide(_engine(), grounded_proposal(invented_ref=True), make_projection())
    assert out.policy.decision == PolicyDecision.ABSTAIN
    assert "R2_grounding" in out.policy.rule_ids


def test_prohibited_term_is_suppressed() -> None:
    engine = _engine(PolicyRuleSet(prohibited_terms=("you have diabetes",)))
    out = _decide(
        engine, grounded_proposal(message="Based on this, you have diabetes."), make_projection()
    )
    assert out.policy.decision == PolicyDecision.SUPPRESSED
    assert "R3_scope" in out.policy.rule_ids


def test_allergen_mention_is_suppressed() -> None:
    proj = make_projection(allergy_substance="penicillin")
    out = _decide(_engine(), grounded_proposal(message="You could take penicillin for this."), proj)
    assert out.policy.decision == PolicyDecision.SUPPRESSED
    assert "R4_allergy" in out.policy.rule_ids


def test_structural_high_severity_event_forces_escalation() -> None:
    out = _decide(_engine(), grounded_proposal(), make_projection(high_severity_event=True))
    assert out.policy.decision == PolicyDecision.APPROVED
    assert out.type == OutputType.FLAG
    assert out.recommended_action in (
        RecommendedAction.SEEK_CARE,
        RecommendedAction.SEEK_URGENT_CARE,
    )
    assert out.escalation.triggered is True
    assert out.evidence  # escalation stays grounded in the PSG event


def test_red_flag_fires_regardless_of_llm_output() -> None:
    # Even an ungrounded/unsafe proposal must not swallow acute-safety escalation.
    out = _decide(
        _engine(),
        grounded_proposal(with_evidence=False, message="everything is fine, ignore it"),
        make_projection(high_severity_event=True),
    )
    assert out.escalation.triggered is True
    assert out.recommended_action == RecommendedAction.SEEK_CARE


def test_configured_red_flag_matches_event_type() -> None:
    ruleset = PolicyRuleSet(
        red_flags=(
            RedFlagRule(
                id="rf_afib",
                action=RecommendedAction.SEEK_URGENT_CARE,
                any_active_event_type=("afib",),
            ),
        )
    )
    proj = make_projection(high_severity_event=False)
    # Inject a non-high event of the matching type via a moderate-severity event.
    from schemas.psg import EventSeverity, EventSummary

    proj.active_events.append(EventSummary(type="afib", severity=EventSeverity.LOW, onset_ts=NOW))
    out = _decide(_engine(ruleset), grounded_proposal(), proj)
    assert out.escalation.triggered is True
    assert out.recommended_action == RecommendedAction.SEEK_URGENT_CARE
    assert "rf_afib" in out.policy.rule_ids


def test_confidence_below_threshold_abstains() -> None:
    ruleset = PolicyRuleSet(confidence_thresholds={OT.INFO: 0.8})
    out = _decide(_engine(ruleset), grounded_proposal(confidence=0.6), make_projection())
    assert out.policy.decision == PolicyDecision.ABSTAIN
    assert "R6_confidence" in out.policy.rule_ids


def test_confidence_gate_off_when_unset() -> None:
    out = _decide(_engine(), grounded_proposal(confidence=0.1), make_projection())
    assert out.policy.decision == PolicyDecision.APPROVED  # no threshold configured


def test_population_fallback_is_downgraded_and_caveated() -> None:
    out = _decide(
        _engine(),
        grounded_proposal(population_fallback=True, confidence=0.9),
        make_projection(population_fallback=True),
    )
    assert out.policy.decision == PolicyDecision.DOWNGRADED
    assert "R7_population_fallback" in out.policy.rule_ids
    assert out.confidence <= 0.5
    assert "general population reference" in out.message


def test_moderate_severity_gets_escalation_path() -> None:
    out = _decide(
        _engine(),
        grounded_proposal(severity=Severity.MODERATE, action=RecommendedAction.NONE),
        make_projection(),
    )
    assert out.recommended_action != RecommendedAction.NONE
