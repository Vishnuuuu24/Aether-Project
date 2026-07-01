"""Output contract validation (docs/04 §6): the mechanical grounding gate and
the mandatory disclaimer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from schemas import (
    MANDATORY_DISCLAIMER,
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
from schemas.output_contract import VersionStamp


def vs() -> VersionStamp:
    return VersionStamp(model="m1", ruleset="r1", baseline_engine="b1", prompt="p1")


def out(**overrides: object) -> OutputContract:
    data: dict[str, object] = {
        "patient_id": uuid4(),
        "type": OutputType.INFO,
        "message": "Your resting heart rate is within your normal range.",
        "severity": Severity.NONE,
        "confidence": 0.9,
        "evidence": [
            Evidence(
                kind=EvidenceKind.PSG_FACT,
                ref="baseline:heart_rate:resting",
                quote_or_fact="resting HR baseline centre 58 bpm",
            )
        ],
        "recommended_action": RecommendedAction.NONE,
        "policy": PolicyRecord(decision=PolicyDecision.APPROVED),
        "versions": vs(),
        "created_at": datetime(2026, 6, 1, tzinfo=UTC),
    }
    data.update(overrides)
    return OutputContract(**data)  # type: ignore[arg-type]


def test_approved_output_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        out(evidence=[])  # ungrounded approved output is a contract violation


def test_approved_with_evidence_is_valid() -> None:
    o = out()
    assert o.policy.decision is PolicyDecision.APPROVED
    assert o.disclaimer == MANDATORY_DISCLAIMER


def test_abstain_is_exempt_from_grounding() -> None:
    o = out(
        evidence=[],
        policy=PolicyRecord(decision=PolicyDecision.ABSTAIN),
        message="I can't answer this from your data.",
    )
    assert o.abstained.value is False  # default; abstain still carries no claims


def test_suppressed_is_exempt_from_grounding() -> None:
    out(evidence=[], policy=PolicyRecord(decision=PolicyDecision.SUPPRESSED))


def test_roundtrip_output_and_proposed() -> None:
    o = out()
    assert OutputContract.model_validate_json(o.model_dump_json()) == o

    proposed = ProposedOutput(
        type=OutputType.INFO,
        message="draft",
        severity=Severity.LOW,
        confidence=0.5,
        evidence=[Evidence(kind=EvidenceKind.KB_PASSAGE, ref="kb:1", quote_or_fact="…")],
    )
    assert ProposedOutput.model_validate_json(proposed.model_dump_json()) == proposed
