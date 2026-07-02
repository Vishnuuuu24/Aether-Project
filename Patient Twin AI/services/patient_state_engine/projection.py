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
    BaselineNode,
    BaselineSummary,
    DeviationNode,
    DeviationSummary,
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
) -> PSGProjection:
    granted = granted_scopes(consent)

    baselines: list[BaselineSummary] = []
    recent_deviations: list[DeviationSummary] = []
    node_times: list[datetime] = []

    if ConsentScope.VITALS in granted:
        current = store.current_baselines(patient_id)
        baselines = [_baseline_summary(n) for n in current]
        devs = store.recent_deviations(patient_id, limit=deviation_limit)
        recent_deviations = [_deviation_summary(d) for d in devs]
        node_times = [n.created_at for n in current] + [d.created_at for d in devs]

    # DOCUMENTS / FORECAST sections are produced in later sprints; scoping them now
    # keeps the projection correct once those nodes exist.

    as_of = max(node_times) if node_times else now
    return PSGProjection(
        patient_age_years=profile.age_years,
        patient_sex_at_birth=profile.sex_at_birth.value,
        baselines=baselines,
        recent_deviations=recent_deviations,
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
