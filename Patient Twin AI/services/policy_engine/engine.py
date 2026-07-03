"""Policy Engine v1 — the deterministic last gate (docs/06 §2-5, docs/10 T4.3).

Given an LLM `ProposedOutput` + the patient's `PSGProjection` + the evidence that was
retrieved, produce the final `OutputContract`. Pure and deterministic: same inputs →
same decision. Side effects (persisting the output, enqueuing a clinician escalation,
writing the audit event) are the orchestrator's job, exactly like EventEngine returns
candidates and the State Engine commits them.

Check order (docs/06 §2). Two structural deviations from the numbered list, both
documented and safety-motivated:

  * Red-flag escalation is evaluated FIRST, not fifth. §2.5 says it fires "regardless
    of LLM output", and §7 says the patient must still receive a safe seek-care message
    even when the rest of the proposal is unusable. Evaluating it after grounding/scope
    would let an upstream suppression swallow an acute-safety escalation — the opposite
    of fail-safe. So acute red flags short-circuit to a deterministic escalation output.
  * The remaining checks then run in order, first hard failure wins:
      1. schema validity      2. grounding        3. scope
      4. allergy/interaction  6. confidence       7. population-fallback honesty

Only this engine may emit `decision == approved`.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from ai.grounding import allowed_refs, psg_facts
from core.versioning.registry import VersionSet
from schemas.output_contract import (
    Abstention,
    Escalation,
    Evidence,
    EvidenceKind,
    OutputContract,
    OutputType,
    PolicyDecision,
    PolicyRecord,
    ProposedOutput,
    RecommendedAction,
    Severity,
)
from schemas.psg import EventSeverity, PSGProjection
from schemas.retrieval import EvidenceChunk

from .rules import PolicyRuleSet, RedFlagRule, severity_rank

POLICY_ENGINE_VERSION = "policy-engine-v1"

_SEVERITY_ORDER = [Severity.NONE, Severity.LOW, Severity.MODERATE, Severity.HIGH]
# A safe, non-diagnostic escalation message. Deterministic — never LLM-authored.
_ESCALATION_MESSAGE = (
    "Your recent readings show a pattern that should be checked by a clinician. "
    "Please seek care. If you have severe or worsening symptoms, contact your local "
    "emergency services."
)


class PolicyEngine:
    def __init__(self, ruleset: PolicyRuleSet, *, version: str = POLICY_ENGINE_VERSION) -> None:
        self._rules = ruleset
        self._version = version

    def decide(
        self,
        proposal: ProposedOutput,
        projection: PSGProjection,
        *,
        patient_id: UUID,
        evidence: list[EvidenceChunk],
        versions: VersionSet,
        now: datetime,
    ) -> OutputContract:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        # (0) Acute red flags — evaluated first, fire regardless of LLM output.
        red_flag = self._match_red_flag(projection)
        if red_flag is not None:
            return self._forced_escalation(red_flag, projection, patient_id, versions, now)

        # (1) Schema validity.
        if not isinstance(proposal, ProposedOutput):
            return self._suppress(
                "output failed schema validation", ["R1_schema"], patient_id, versions, now
            )

        # (2) Grounding — mechanical anti-hallucination gate.
        allowed = allowed_refs(projection, evidence)
        cited = {e.ref for e in proposal.evidence}
        if not proposal.evidence:
            return self._abstain(
                "insufficient grounded evidence to answer safely",
                ["R2_grounding"],
                patient_id,
                versions,
                now,
            )
        if not cited.issubset(allowed):
            return self._abstain(
                "proposed answer cited evidence that was not provided (possible hallucination)",
                ["R2_grounding"],
                patient_id,
                versions,
                now,
            )

        # (3) Scope — closed action vocabulary is enum-guaranteed; scan the prohibited
        #     clinical lexicon (config, UNSET => inert).
        message_lc = proposal.message.lower()
        hit = next((t for t in self._rules.prohibited_terms if t in message_lc), None)
        if hit is not None:
            return self._suppress(
                "message contains out-of-scope clinical language",
                ["R3_scope"],
                patient_id,
                versions,
                now,
            )

        # (4) Allergy / interaction — mention of a known allergen is suppressed + flagged.
        allergen = next(
            (
                a.substance
                for a in projection.allergies
                if a.substance and a.substance.lower() in message_lc
            ),
            None,
        )
        if allergen is not None:
            return self._suppress(
                f"message references a substance the patient is allergic to ({allergen})",
                ["R4_allergy"],
                patient_id,
                versions,
                now,
            )

        # (6) Confidence threshold (per output type; UNSET => gate off).
        threshold = self._rules.confidence_thresholds.get(proposal.type)
        if threshold is not None and proposal.confidence < threshold:
            return self._abstain(
                f"confidence {proposal.confidence:.2f} below threshold "
                f"{threshold:.2f} for {proposal.type.value}",
                ["R6_confidence"],
                patient_id,
                versions,
                now,
            )

        # (7) Population-fallback honesty — a population baseline may not be presented
        #     as the patient's personalised normal.
        rule_ids = ["R2_grounding"]
        decision = PolicyDecision.APPROVED
        message = proposal.message
        confidence = proposal.confidence
        uses_fallback = (
            proposal.baseline_reference is not None
            and proposal.baseline_reference.is_population_fallback
        )
        if uses_fallback:
            # A population baseline is never presented as the patient's personalised
            # normal — downgrade, cap confidence, and always attach the caveat (whether
            # or not the model already used personalised phrasing).
            decision = PolicyDecision.DOWNGRADED
            rule_ids.append("R7_population_fallback")
            confidence = min(confidence, 0.5)
            message = (
                f"{message} Note: this comparison uses a general population reference, "
                "not a baseline personalised to you yet."
            )

        # Mandatory escalation path when severity is at least moderate (docs/06 §4).
        action = proposal.recommended_action
        if severity_rank_from_output(proposal.severity) >= severity_rank(EventSeverity.MODERATE):
            if action == RecommendedAction.NONE:
                action = RecommendedAction.MONITOR

        return OutputContract(
            patient_id=patient_id,
            type=proposal.type,
            message=message,
            severity=proposal.severity,
            confidence=confidence,
            evidence=list(proposal.evidence),
            baseline_reference=proposal.baseline_reference,
            recommended_action=action,
            escalation=Escalation(triggered=False),
            abstained=Abstention(value=False),
            policy=PolicyRecord(decision=decision, rule_ids=rule_ids),
            versions=versions.output_stamp(),
            created_at=now,
        )

    def on_gateway_failure(
        self,
        reason: str,
        *,
        projection: PSGProjection,
        patient_id: UUID,
        versions: VersionSet,
        now: datetime,
    ) -> OutputContract:
        """Deterministic outcome when the LLM Gateway fails or blocks egress
        (docs/06 §9). The copilot routes gateway errors through here so EVERY
        user-facing output still originates from the Policy Engine and carries a
        decision record.

        Acute safety is independent of the LLM: a red flag in the patient's state must
        still escalate even with the gateway dead (docs/06 §7). So we evaluate red
        flags first here too, and only abstain if none fire. Never falls back to
        ungrounded generation.
        """
        red_flag = self._match_red_flag(projection)
        if red_flag is not None:
            return self._forced_escalation(red_flag, projection, patient_id, versions, now)
        return self._abstain(reason, ["R0_gateway_unavailable"], patient_id, versions, now)

    # -- red-flag matching -------------------------------------------------------

    def _match_red_flag(self, projection: PSGProjection) -> RedFlagRule | None:
        # Structural rule (always on): any HIGH-severity active event — that severity
        # was assigned by the Event Engine's clinician-defined rules, so escalating on
        # it invents no new threshold.
        if any(e.severity == EventSeverity.HIGH for e in projection.active_events):
            return RedFlagRule(
                id="R5_structural_high_severity_event", action=RecommendedAction.SEEK_CARE
            )

        # Configured acute patterns (UNSET => none).
        for rule in self._rules.red_flags:
            if self._red_flag_matches(rule, projection):
                return rule
        return None

    @staticmethod
    def _red_flag_matches(rule: RedFlagRule, projection: PSGProjection) -> bool:
        events = projection.active_events
        if rule.any_active_event_type:
            events = [e for e in events if e.type in rule.any_active_event_type]
        if rule.min_event_severity is not None:
            floor = severity_rank(rule.min_event_severity)
            events = [e for e in events if severity_rank(e.severity) >= floor]
            return bool(events)
        # No severity floor but a type filter given => any matching-type event fires.
        return bool(events) if rule.any_active_event_type else False

    # -- output builders ---------------------------------------------------------

    def _forced_escalation(
        self,
        rule: RedFlagRule,
        projection: PSGProjection,
        patient_id: UUID,
        versions: VersionSet,
        now: datetime,
    ) -> OutputContract:
        facts = psg_facts(projection)
        evidence = [
            Evidence(kind=EvidenceKind.PSG_FACT, ref=ref, quote_or_fact=text)
            for ref, text in facts.items()
            if ref.startswith("psg:event:")
        ]
        if not evidence and facts:  # fall back to any PSG fact so the output stays grounded
            ref, text = next(iter(facts.items()))
            evidence = [Evidence(kind=EvidenceKind.PSG_FACT, ref=ref, quote_or_fact=text)]
        return OutputContract(
            patient_id=patient_id,
            type=OutputType.FLAG,
            message=_ESCALATION_MESSAGE,
            severity=Severity.HIGH,
            confidence=1.0,  # deterministic policy decision, not a model estimate
            evidence=evidence,
            recommended_action=rule.action,
            escalation=Escalation(triggered=True, reason=f"red-flag rule {rule.id}"),
            abstained=Abstention(value=False),
            policy=PolicyRecord(decision=PolicyDecision.APPROVED, rule_ids=[rule.id]),
            versions=versions.output_stamp(),
            created_at=now,
        )

    def _abstain(
        self,
        reason: str,
        rule_ids: list[str],
        patient_id: UUID,
        versions: VersionSet,
        now: datetime,
    ) -> OutputContract:
        return OutputContract(
            patient_id=patient_id,
            type=OutputType.INFO,
            message=f"I can't answer this safely from the available data. {reason}. "
            "Consider discussing this with your clinician.",
            severity=Severity.NONE,
            confidence=0.0,
            evidence=[],
            recommended_action=RecommendedAction.NONE,
            abstained=Abstention(value=True, reason=reason),
            policy=PolicyRecord(decision=PolicyDecision.ABSTAIN, rule_ids=rule_ids),
            versions=versions.output_stamp(),
            created_at=now,
        )

    def _suppress(
        self,
        reason: str,
        rule_ids: list[str],
        patient_id: UUID,
        versions: VersionSet,
        now: datetime,
    ) -> OutputContract:
        return OutputContract(
            patient_id=patient_id,
            type=OutputType.INFO,
            message="This response was withheld by a safety check. "
            "Consider discussing your question with your clinician.",
            severity=Severity.NONE,
            confidence=0.0,
            evidence=[],
            recommended_action=RecommendedAction.NONE,
            abstained=Abstention(value=True, reason=reason),
            policy=PolicyRecord(decision=PolicyDecision.SUPPRESSED, rule_ids=rule_ids),
            versions=versions.output_stamp(),
            created_at=now,
        )


def severity_rank_from_output(severity: Severity) -> int:
    return _SEVERITY_ORDER.index(severity)
