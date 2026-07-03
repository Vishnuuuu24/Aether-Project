"""Deterministic fixtures for Policy Engine tests."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from core.versioning.registry import VersionSet
from schemas.output_contract import (
    BaselineReference,
    Evidence,
    EvidenceKind,
    OutputType,
    ProposedOutput,
    RecommendedAction,
    Severity,
)
from schemas.psg import (
    AllergySummary,
    BaselineSummary,
    DeviationDirection,
    DeviationSummary,
    EventSeverity,
    EventSummary,
    PSGProjection,
    VersionStamp,
)
from schemas.reading import MeasurementContext, MetricCode
from schemas.retrieval import EvidenceChunk
from schemas.vector import VectorSourceType

NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
PATIENT_ID = UUID("22222222-2222-2222-2222-222222222222")
KB_CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")

VERSIONS = VersionSet(
    model="m1", ruleset="policy-unset", prompt="p1", baseline_engine="stat-v1", schema="s1"
)


def kb_evidence() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id=KB_CHUNK_ID,
            source_type=VectorSourceType.KB_PASSAGE,
            text="Resting heart rate rises transiently with acute stress or illness.",
            score=0.9,
        )
    ]


def make_projection(
    *,
    high_severity_event: bool = False,
    event_type: str = "generic_event",
    allergy_substance: str | None = None,
    population_fallback: bool = False,
) -> PSGProjection:
    events: list[EventSummary] = []
    if high_severity_event:
        events.append(EventSummary(type=event_type, severity=EventSeverity.HIGH, onset_ts=NOW))
    allergies: list[AllergySummary] = []
    if allergy_substance is not None:
        allergies.append(
            AllergySummary(substance=allergy_substance, reaction="hives", severity="moderate")
        )
    return PSGProjection(
        patient_age_years=41,
        patient_sex_at_birth="female",
        baselines=[
            BaselineSummary(
                metric_code=MetricCode.HEART_RATE,
                context=MeasurementContext.RESTING,
                center=58.0,
                dispersion=4.0,
                confidence=0.9,
                is_population_fallback=population_fallback,
            )
        ],
        recent_deviations=[
            DeviationSummary(
                metric_code=MetricCode.HEART_RATE,
                direction=DeviationDirection.UP,
                magnitude=12.0,
                z_robust=3.1,
                confidence=0.8,
                ts=NOW,
            )
        ],
        active_events=events,
        allergies=allergies,
        as_of=NOW,
        consent_scope=["copilot", "vitals"],
        versions=VersionStamp(baseline_engine="stat-v1", ruleset="unset", prompt="p1", model="m1"),
    )


def grounded_proposal(
    *,
    message: str = "Your resting heart rate rose above your usual range overnight.",
    severity: Severity = Severity.LOW,
    confidence: float = 0.7,
    action: RecommendedAction = RecommendedAction.MONITOR,
    output_type: OutputType = OutputType.INFO,
    with_evidence: bool = True,
    population_fallback: bool = False,
    invented_ref: bool = False,
) -> ProposedOutput:
    evidence: list[Evidence] = []
    if with_evidence:
        ref = "kb:deadbeef" if invented_ref else f"kb:{KB_CHUNK_ID}"
        evidence.append(
            Evidence(kind=EvidenceKind.KB_PASSAGE, ref=ref, quote_or_fact="RHR rises with stress.")
        )
    baseline_ref = None
    if population_fallback:
        baseline_ref = BaselineReference(
            metric_code="heart_rate", center=60.0, dispersion=5.0, is_population_fallback=True
        )
    return ProposedOutput(
        type=output_type,
        message=message,
        severity=severity,
        confidence=confidence,
        evidence=evidence,
        baseline_reference=baseline_ref,
        recommended_action=action,
    )
