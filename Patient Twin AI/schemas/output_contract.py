"""Output contract: every user-facing output, post-Policy, must be an
instance of this model. Source of truth: docs/04 §6, docs/06.

The Policy Engine is the only thing allowed to produce an OutputContract
with policy.decision == "approved". Everything else (LLM Gateway proposals,
drafts) must use ProposedOutput and never reach a client directly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class OutputType(str, Enum):
    INFO = "info"
    FLAG = "flag"
    GUIDANCE = "guidance"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class RecommendedAction(str, Enum):
    """Closed vocabulary. No free-text actions, ever."""

    NONE = "none"
    MONITOR = "monitor"
    LIFESTYLE_INFO = "lifestyle_info"
    SEEK_CARE = "seek_care"
    SEEK_URGENT_CARE = "seek_urgent_care"


class EvidenceKind(str, Enum):
    PSG_FACT = "psg_fact"
    KB_PASSAGE = "kb_passage"


class Evidence(BaseModel):
    kind: EvidenceKind
    ref: str
    quote_or_fact: str


class Escalation(BaseModel):
    triggered: bool = False
    reason: str | None = None


class Abstention(BaseModel):
    value: bool = False
    reason: str | None = None


class PolicyDecision(str, Enum):
    APPROVED = "approved"
    DOWNGRADED = "downgraded"
    SUPPRESSED = "suppressed"
    ABSTAIN = "abstain"


class PolicyRecord(BaseModel):
    decision: PolicyDecision
    rule_ids: list[str] = Field(default_factory=list)


class VersionStamp(BaseModel):
    model: str
    ruleset: str
    baseline_engine: str
    prompt: str


class BaselineReference(BaseModel):
    metric_code: str
    center: float
    dispersion: float
    is_population_fallback: bool


MANDATORY_DISCLAIMER = (
    "This is not a doctor and not an emergency service. "
    "For emergencies, contact your local emergency services immediately."
)


class ProposedOutput(BaseModel):
    """What the LLM Gateway emits. NEVER shown to a user directly.

    Must pass through the Policy Engine, which turns this into an
    OutputContract (or suppresses/downgrades/forces abstain).
    """

    type: OutputType
    message: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence]
    baseline_reference: BaselineReference | None = None
    recommended_action: RecommendedAction = RecommendedAction.NONE


class OutputContract(BaseModel):
    """Final, Policy-approved, user-facing output. docs/04 §6."""

    output_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID
    type: OutputType
    message: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence]
    baseline_reference: BaselineReference | None = None
    recommended_action: RecommendedAction
    escalation: Escalation = Field(default_factory=Escalation)
    abstained: Abstention = Field(default_factory=Abstention)
    policy: PolicyRecord
    disclaimer: str = MANDATORY_DISCLAIMER
    versions: VersionStamp
    created_at: datetime

    @model_validator(mode="after")
    def grounding_required(self) -> OutputContract:
        """Mechanical anti-hallucination gate (docs/06 §2.2).

        If the message contains substantive content but there is no
        evidence at all, this is a contract violation. Abstention/
        suppression outputs are exempt (they carry no claims).
        """
        if (
            self.policy.decision not in (PolicyDecision.ABSTAIN, PolicyDecision.SUPPRESSED)
            and not self.evidence
        ):
            raise ValueError(
                "approved/downgraded outputs must carry at least one evidence "
                "ref — ungrounded claims are a contract violation, not a style choice"
            )
        return self
