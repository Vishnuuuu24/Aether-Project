"""End-to-end pipeline integration (docs/15 T6.1; docs/11 §2 rung 2).

One synthetic patient is driven through the *production* engines, in order, sharing
ONE audit chain:

    ingest (normalise + consent) → SQI gate + baseline learn → deviation scoring
    → PSG commit (baseline + deviation) → event detection + commit
    → forecast + commit → consent-scoped projection → copilot answer (policy-gated)

The seams are the real ones — no engine is mocked. Only the two *external*
dependencies the copilot talks to are stubbed: the LLM itself (‹GPU-DEP›, never run
on the Mac per CLAUDE.md) and the vector store behind the retriever. Both return
deterministic, grounded fixtures so the Policy Engine can reach a real APPROVED
decision instead of abstaining for lack of an LLM.

Asserted:
  * the final `OutputContract` is Policy-**approved** (grounded path) / a valid
    forced-escalation (red-flag path);
  * every stage wrote to the single audit chain and `verify_chain` holds;
  * the copilot's output is reconstructable from that chain (its output_id appears
    in an audit record) — the docs/11 rung-2 "audit reconstruction" requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from ai.baseline.statistical import StatisticalBaselineEngine
from ai.features.sqi import SqiGate
from ai.forecasting.holt import HoltLinearForecaster
from core.audit import AuditWriter, InMemoryAuditStore, verify_chain
from core.versioning import VersionSet
from schemas.audit import AuditAction, AuditRecord
from schemas.baseline import DeviationMagnitude
from schemas.consent import Consent, ConsentScope
from schemas.forecast import MetricSeries, SeriesPoint
from schemas.output_contract import OutputContract, PolicyDecision, ProposedOutput
from schemas.patient import PatientProfile, SexAtBirth
from schemas.psg import DeviationDirection, EventSeverity
from schemas.reading import MeasurementContext, MetricCode
from schemas.retrieval import EvidenceChunk, RetrievalScope
from services.copilot_service.audit import AuditWriterSink
from services.copilot_service.orchestrator import Copilot
from services.event_engine.engine import EventEngine
from services.event_engine.rules import CoOccurrenceRule, EventRuleSet, MetricCondition
from services.ingestion_service.consent import StaticConsentProvider
from services.ingestion_service.service import IngestionService
from services.ingestion_service.sink import InMemoryReadingSink
from services.patient_state_engine.profile import StaticProfileProvider
from services.patient_state_engine.service import PatientStateEngine
from services.patient_state_engine.store import InMemoryPSGStore
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import PolicyRuleSet
from services.policy_engine.tests._fixtures import grounded_proposal, kb_evidence

# One version set stamped by every stage — realistic single-release wiring.
VERSIONS = VersionSet(
    model="m1", ruleset="r1", prompt="p1", baseline_engine="statistical-v1", schema="s1"
)
BASE = datetime(2026, 6, 1, tzinfo=UTC)
OCCURRED_AT = datetime(2026, 6, 21, 8, 0, tzinfo=UTC)
NOW = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)

# SQI thresholds are clinical config (UNSET in prod); the mechanism is exercised with
# an explicit test threshold so quality-passing readings can enter the baseline.
SQI_THRESHOLDS = {MetricCode.HEART_RATE.value: 0.5}

# Values chosen so 60 readings give median 60 and MAD 2 → robust sigma ≈ 2.965.
_NORMAL_CYCLE = [56.0, 58.0, 60.0, 62.0, 64.0]
_N_NORMAL = 60  # ≥ min_n=50, packed 3/day over 20 days → span ≥ min_days=7, one 28d window


# -- external-dependency stubs (LLM + vector store only) --------------------------


class _StubRetriever:
    """Stands in for the vector store; returns a fixed grounded KB chunk."""

    def __init__(self, evidence: list[EvidenceChunk]) -> None:
        self._evidence = evidence
        self.last_scope: RetrievalScope | None = None

    def search(self, query: str, scope: RetrievalScope, *, k: int = 10) -> list[EvidenceChunk]:
        self.last_scope = scope
        return self._evidence


class _StubGateway:
    """Stands in for the ‹GPU-DEP› LLM; returns a fixed grounded proposal."""

    def __init__(self, proposal: ProposedOutput) -> None:
        self._proposal = proposal

    def propose(
        self,
        *,
        query: str,
        projection: object,
        evidence: list[EvidenceChunk],
        locale: str = "en",
    ) -> ProposedOutput:
        return self._proposal


@dataclass
class _Recorder:
    """Collects copilot side effects (output store + escalation queue)."""

    saved: list[OutputContract] = field(default_factory=list)
    escalated: list[OutputContract] = field(default_factory=list)

    def save(self, output: OutputContract) -> None:
        self.saved.append(output)

    def enqueue(self, output: OutputContract) -> None:
        self.escalated.append(output)


# -- synthetic data ---------------------------------------------------------------


def _normal_reading_dicts(patient_id: UUID) -> list[dict[str, object]]:
    """60 resting-HR readings, 3/day (07:00/08:00/09:00 → all 'morning' bucket) over
    20 days: within one 28-day window, spanning ≥ 7 days, ≥ min_n. Median 60, MAD 2.
    """
    out: list[dict[str, object]] = []
    for i in range(_N_NORMAL):
        day, slot = divmod(i, 3)
        ts = BASE + timedelta(days=day, hours=7 + slot)
        out.append(
            {
                "patient_id": patient_id,
                "metric_code": MetricCode.HEART_RATE.value,
                "value": _NORMAL_CYCLE[i % len(_NORMAL_CYCLE)],
                "unit": "bpm",
                "timestamp": ts,
                "source_device": "apple_watch_s9",
                "sqi": 0.9,
                "context": MeasurementContext.RESTING.value,
                "ingest_adapter": "csv",
            }
        )
    return out


def _anomaly_reading_dict(patient_id: UUID, value: float) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "metric_code": MetricCode.HEART_RATE.value,
        "value": value,
        "unit": "bpm",
        "timestamp": OCCURRED_AT,  # day 20, 08:00 → same 'morning' bucket
        "source_device": "apple_watch_s9",
        "sqi": 0.9,
        "context": MeasurementContext.RESTING.value,
        "ingest_adapter": "csv",
    }


def _event_ruleset() -> EventRuleSet:
    """A single co-occurrence rule (mechanism only — no clinical content): resting HR
    deviating UP by at least a MODERATE bucket, a single reading sufficient.
    """
    return EventRuleSet(
        rules=(
            CoOccurrenceRule(
                id="it-hr-up",
                event_type="physiological_stress/possible_illness",
                window_minutes=24 * 60,
                persistence_count=1,
                conditions=(
                    MetricCondition(
                        metric_code=MetricCode.HEART_RATE.value,
                        direction=DeviationDirection.UP,
                        min_magnitude=DeviationMagnitude.MODERATE,
                    ),
                ),
            ),
        ),
        version="integration-test",
    )


# -- the pipeline -----------------------------------------------------------------


@dataclass
class _PipelineRun:
    output: OutputContract
    audit_store: InMemoryAuditStore
    recorder: _Recorder
    retriever: _StubRetriever
    deviation_magnitude: DeviationMagnitude
    event_severity: EventSeverity


def _run_pipeline(anomaly_value: float) -> _PipelineRun:
    patient_id = uuid4()

    # ONE audit chain, shared by ingestion, the state engine, and the copilot.
    audit_store = InMemoryAuditStore()
    audit_writer = AuditWriter(audit_store)

    consent = Consent(
        scope=[ConsentScope.VITALS, ConsentScope.FORECAST, ConsentScope.COPILOT],
        version="v1",
        granted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    consent_provider = StaticConsentProvider()
    consent_provider.grant(patient_id, consent)

    profiles = StaticProfileProvider()
    profiles.put(
        PatientProfile(
            patient_id=patient_id,
            consent=consent,
            age_years=41,
            sex_at_birth=SexAtBirth.FEMALE,
        )
    )

    # --- Stage 1: ingestion (normalise + consent gate + audit) ---
    sink = InMemoryReadingSink()
    ingestion = IngestionService(
        consent_provider=consent_provider, sink=sink, audit_writer=audit_writer
    )
    raw = _normal_reading_dicts(patient_id) + [_anomaly_reading_dict(patient_id, anomaly_value)]
    ingest_result = ingestion.ingest(raw, adapter="csv")
    assert len(ingest_result.accepted) == len(raw)
    assert not ingest_result.rejected
    readings = sink.readings
    normal_readings, anomaly_reading = readings[:_N_NORMAL], readings[_N_NORMAL]

    # --- Stage 2: SQI gate + baseline learning, then deviation scoring ---
    baseline_engine = StatisticalBaselineEngine(gate=SqiGate(SQI_THRESHOLDS))
    for reading in normal_readings:
        baseline_engine.update(reading)
    deviation = baseline_engine.score(anomaly_reading)
    baseline = baseline_engine.get_baseline(
        MetricCode.HEART_RATE.value, MeasurementContext.RESTING.value
    )

    # --- Stage 3: PSG commit (baseline + deviation), append-only + audited ---
    state = PatientStateEngine(
        store=InMemoryPSGStore(),
        consent_provider=consent_provider,
        audit_writer=audit_writer,
        versions=VERSIONS,
        profile_provider=profiles,
        clock=lambda: OCCURRED_AT,
    )
    commit = state.commit_deviation(baseline, deviation, occurred_at=OCCURRED_AT)
    assert commit.baseline_committed  # a personalised baseline version was written
    assert commit.deviation_node is not None  # the anomaly actually deviated

    # --- Stage 4: event detection + commit ---
    events = EventEngine(_event_ruleset()).evaluate(
        patient_id, [commit.deviation_node], as_of=OCCURRED_AT
    )
    assert len(events) == 1
    event_node = state.commit_event(events[0], occurred_at=OCCURRED_AT)

    # --- Stage 5: forecast + commit ---
    series = MetricSeries(
        patient_id=patient_id,
        metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING,
        points=[
            SeriesPoint(ts=BASE + timedelta(days=d), value=_NORMAL_CYCLE[d % len(_NORMAL_CYCLE)])
            for d in range(20)
        ],
    )
    forecast = HoltLinearForecaster().forecast(series, horizon_days=3)
    state.commit_forecast(forecast, generated_at=OCCURRED_AT)

    # --- Stage 6: consent-scoped projection (no raw signals) ---
    projection = state.build_projection(patient_id)
    assert projection.baselines and projection.recent_deviations
    assert projection.active_events and projection.latest_forecasts

    # --- Stage 7: copilot answer (retrieve → propose → policy → persist/audit) ---
    recorder = _Recorder()
    retriever = _StubRetriever(kb_evidence())
    copilot = Copilot(
        retriever=retriever,
        gateway=_StubGateway(grounded_proposal()),
        policy=PolicyEngine(PolicyRuleSet()),
        versions=VERSIONS,
        output_store=recorder,
        escalation_sink=recorder,
        audit_sink=AuditWriterSink(audit_writer),  # same chain as every other stage
    )
    output = copilot.answer(
        patient_id=patient_id,
        projection=projection,
        query="why is my resting heart rate up?",
        consented_scopes=[ConsentScope.COPILOT, ConsentScope.VITALS],
        now=NOW,
    )

    return _PipelineRun(
        output=output,
        audit_store=audit_store,
        recorder=recorder,
        retriever=retriever,
        deviation_magnitude=deviation.magnitude,
        event_severity=event_node.severity,
    )


# -- assertions -------------------------------------------------------------------


def _output_reconstructable(records: list[AuditRecord], output: OutputContract) -> bool:
    """The copilot's output must be recoverable from the audit chain (docs/11 rung 2)."""
    return any(str(output.output_id) in r.output_refs for r in records)


def test_grounded_path_end_to_end_is_policy_approved_and_audit_reconstructs() -> None:
    # A MODERATE anomaly (z ≈ 3.4) raises a MODERATE event — no red flag — so the copilot
    # runs the full grounded path and the Policy Engine reaches a real APPROVED decision.
    run = _run_pipeline(anomaly_value=70.0)

    assert run.deviation_magnitude is DeviationMagnitude.MODERATE
    assert run.event_severity is EventSeverity.MODERATE

    assert run.output.policy.decision is PolicyDecision.APPROVED
    assert run.output.abstained.value is False
    assert run.output.evidence  # an approved answer is grounded
    assert run.output.escalation.triggered is False
    assert run.recorder.saved == [run.output]  # persisted exactly once
    assert run.recorder.escalated == []  # nothing to escalate

    # Retrieval was consent-scoped to the patient (docs/07 §5).
    assert run.retriever.last_scope is not None
    assert ConsentScope.COPILOT in run.retriever.last_scope.consented_scopes

    # One shared, intact chain covering every mutation, first→last stage.
    records = run.audit_store.records
    verify_chain(records)  # raises AuditChainError on any tampering
    actions = [r.action for r in records]
    assert actions == [
        AuditAction.INGEST,
        AuditAction.BASELINE_UPDATE,
        AuditAction.STATE_COMMIT,  # deviation
        AuditAction.STATE_COMMIT,  # event
        AuditAction.STATE_COMMIT,  # forecast
        AuditAction.POLICY_DECISION,  # copilot output
    ]
    assert _output_reconstructable(records, run.output)


def test_red_flag_path_end_to_end_forces_escalation_and_audit_reconstructs() -> None:
    # A MARKED anomaly (z ≈ 10) raises a HIGH-severity event; acute safety short-circuits
    # to a deterministic forced escalation — independent of the LLM — still fully audited.
    run = _run_pipeline(anomaly_value=90.0)

    assert run.deviation_magnitude is DeviationMagnitude.MARKED
    assert run.event_severity is EventSeverity.HIGH

    # Forced escalation is a valid Policy-approved output (docs/06 §7).
    assert run.output.policy.decision is PolicyDecision.APPROVED
    assert run.output.escalation.triggered is True
    assert run.output.recommended_action.value == "seek_care"
    assert run.recorder.escalated == [run.output]  # queued for clinician review

    records = run.audit_store.records
    verify_chain(records)
    assert [r.action for r in records] == [
        AuditAction.INGEST,
        AuditAction.BASELINE_UPDATE,
        AuditAction.STATE_COMMIT,
        AuditAction.STATE_COMMIT,
        AuditAction.STATE_COMMIT,
        AuditAction.POLICY_DECISION,
    ]
    assert _output_reconstructable(records, run.output)
