"""Consent-scoped PSG projection builder (docs/04 §5, CLAUDE.md principle 2).

The projection is the ONLY structure the LLM may receive. It carries no raw
signals and no reading-level data — only summaries. Sections outside the patient's
granted consent scopes are omitted (left unpopulated), never filled with data the
patient did not consent to disclose.

Scope → section map (docs/02 §2, docs/04 §5):
  VITALS    → baselines, recent_deviations, active_events
  DOCUMENTS → conditions, medications, allergies, observations
  FORECAST  → latest_forecasts
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from core.auth.consent_gate import granted_scopes
from core.versioning import VersionSet
from schemas.consent import Consent, ConsentScope
from schemas.patient import PatientProfile
from schemas.psg import (
    AllergyNode,
    AllergySummary,
    BaselineNode,
    BaselineSummary,
    ConditionNode,
    ConditionSummary,
    DeviationNode,
    DeviationSummary,
    EventNode,
    EventSummary,
    ForecastNode,
    ForecastSummary,
    MedicationNode,
    MedicationSummary,
    ObservationNode,
    ObservationSummary,
    PSGProjection,
)

from .store import PSGStore


def build_projection(
    *,
    patient_id: UUID,
    store: PSGStore,
    consent: Consent | None,
    profile: PatientProfile,
    versions: VersionSet,
    now: datetime,
    deviation_limit: int = 20,
    observation_limit: int = 20,
) -> PSGProjection:
    granted = granted_scopes(consent)

    baselines: list[BaselineSummary] = []
    recent_deviations: list[DeviationSummary] = []
    active_events: list[EventSummary] = []
    latest_forecasts: list[ForecastSummary] = []
    conditions: list[ConditionSummary] = []
    medications: list[MedicationSummary] = []
    allergies: list[AllergySummary] = []
    recent_observations: list[ObservationSummary] = []
    node_times: list[datetime] = []

    if ConsentScope.VITALS in granted:
        current = store.current_baselines(patient_id)
        baselines = [_baseline_summary(n) for n in current]
        devs = store.recent_deviations(patient_id, limit=deviation_limit)
        recent_deviations = [_deviation_summary(d) for d in devs]
        events = store.active_events(patient_id)
        active_events = [_event_summary(e) for e in events]
        node_times += (
            [n.created_at for n in current]
            + [d.created_at for d in devs]
            + [e.created_at for e in events]
        )

    if ConsentScope.FORECAST in granted:
        forecasts = store.latest_forecasts(patient_id)
        latest_forecasts = [_forecast_summary(f) for f in forecasts]
        node_times += [f.created_at for f in forecasts]

    if ConsentScope.DOCUMENTS in granted:
        cond = store.current_conditions(patient_id)
        med = store.current_medications(patient_id)
        alg = store.current_allergies(patient_id)
        obs = store.recent_observations(patient_id, limit=observation_limit)
        conditions = [_condition_summary(c) for c in cond]
        medications = [_medication_summary(m) for m in med]
        allergies = [_allergy_summary(a) for a in alg]
        recent_observations = [_observation_summary(o) for o in obs]
        node_times += (
            [c.created_at for c in cond]
            + [m.created_at for m in med]
            + [a.created_at for a in alg]
            + [o.created_at for o in obs]
        )

    as_of = max(node_times) if node_times else now
    return PSGProjection(
        patient_age_years=profile.age_years,
        patient_sex_at_birth=profile.sex_at_birth.value,
        baselines=baselines,
        recent_deviations=recent_deviations,
        active_events=active_events,
        latest_forecasts=latest_forecasts,
        conditions=conditions,
        medications=medications,
        allergies=allergies,
        recent_observations=recent_observations,
        as_of=as_of,
        consent_scope=sorted(s.value for s in granted),
        versions=versions.projection_stamp(),
    )


def _baseline_summary(node: BaselineNode) -> BaselineSummary:
    return BaselineSummary(
        metric_code=node.metric_code,
        context=node.context,
        center=node.center,
        dispersion=node.dispersion,
        confidence=node.confidence,
        is_population_fallback=node.is_population_fallback,
    )


def _deviation_summary(node: DeviationNode) -> DeviationSummary:
    return DeviationSummary(
        metric_code=node.metric_code,
        direction=node.direction,
        magnitude=node.magnitude,
        z_robust=node.z_robust,
        confidence=node.confidence,
        ts=node.created_at,
    )


def _event_summary(node: EventNode) -> EventSummary:
    return EventSummary(type=node.type, severity=node.severity, onset_ts=node.onset_ts)


def _forecast_summary(node: ForecastNode) -> ForecastSummary:
    return ForecastSummary(
        metric_code=node.metric_code,
        horizon_days=node.horizon_days,
        points=node.points,
        intervals=node.intervals,
    )


def _condition_summary(node: ConditionNode) -> ConditionSummary:
    return ConditionSummary(snomed_code=node.snomed_code, display=node.display, status=node.status)


def _medication_summary(node: MedicationNode) -> MedicationSummary:
    return MedicationSummary(rxnorm_code=node.rxnorm_code, display=node.display, status=node.status)


def _allergy_summary(node: AllergyNode) -> AllergySummary:
    return AllergySummary(
        substance=node.substance_code, reaction=node.reaction, severity=node.severity
    )


def _observation_summary(node: ObservationNode) -> ObservationSummary:
    return ObservationSummary(
        loinc_code=node.loinc_code,
        display=node.display,
        value=node.value,
        unit=node.unit,
        ts=node.ts,
    )
