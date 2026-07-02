"""Shared test builders for baselines, deviations, consent, and a wired engine."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from core.audit import AuditWriter, InMemoryAuditStore
from core.versioning import VersionSet
from schemas.baseline import (
    Baseline,
    BaselineAvailability,
    DeviationMagnitude,
    DeviationResult,
)
from schemas.consent import Consent, ConsentScope
from schemas.patient import PatientProfile, SexAtBirth
from schemas.psg import DeviationDirection
from schemas.reading import MeasurementContext, MetricCode
from services.patient_state_engine.consent import StaticConsentProvider
from services.patient_state_engine.profile import StaticProfileProvider
from services.patient_state_engine.service import PatientStateEngine
from services.patient_state_engine.store import InMemoryPSGStore

OCCURRED_AT = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)

VERSIONS = VersionSet(
    model="m1", ruleset="r1", prompt="p1", baseline_engine="statistical-v1", schema="s1"
)


def baseline(patient_id: UUID, **overrides: Any) -> Baseline:
    data: dict[str, Any] = {
        "patient_id": patient_id,
        "metric_code": MetricCode.HEART_RATE,
        "context": MeasurementContext.RESTING,
        "availability": BaselineAvailability.PERSONALISED,
        "center": 60.0,
        "dispersion_sigma": 1.4826,
        "ewma": 60.0,
        "sample_n": 60,
        "span_days": 20.0,
        "window_days": 28,
        "min_n": 50,
        "min_days": 7,
        "is_population_fallback": False,
        "baseline_engine_version": "statistical-v1",
    }
    data.update(overrides)
    return Baseline(**data)


def fallback_baseline(patient_id: UUID, **overrides: Any) -> Baseline:
    return baseline(
        patient_id,
        availability=BaselineAvailability.POPULATION_FALLBACK,
        center=80.0,
        dispersion_sigma=10.0,
        ewma=None,
        sample_n=1,
        span_days=0.0,
        is_population_fallback=True,
        **overrides,
    )


def deviation(
    patient_id: UUID,
    *,
    magnitude: DeviationMagnitude = DeviationMagnitude.MARKED,
    direction: DeviationDirection = DeviationDirection.UP,
    z_robust: float = 6.0,
    availability: BaselineAvailability = BaselineAvailability.PERSONALISED,
    is_population_fallback: bool = False,
    **overrides: Any,
) -> DeviationResult:
    data: dict[str, Any] = {
        "reading_id": uuid4(),
        "patient_id": patient_id,
        "metric_code": MetricCode.HEART_RATE,
        "context": MeasurementContext.RESTING,
        "z_robust": z_robust,
        "direction": direction,
        "magnitude": magnitude,
        "confidence": 0.8,
        "is_population_fallback": is_population_fallback,
        "baseline_availability": availability,
    }
    data.update(overrides)
    return DeviationResult(**data)


def vitals_consent() -> Consent:
    return Consent(
        scope=[ConsentScope.VITALS],
        version="v1",
        granted_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def profile(patient_id: UUID, **overrides: Any) -> PatientProfile:
    data: dict[str, Any] = {
        "patient_id": patient_id,
        "consent": vitals_consent(),
        "age_years": 40,
        "sex_at_birth": SexAtBirth.MALE,
    }
    data.update(overrides)
    return PatientProfile(**data)


def wired_engine(
    patient_id: UUID,
    *,
    consent: Consent | None = None,
    seed_profile: bool = True,
) -> tuple[PatientStateEngine, InMemoryPSGStore, InMemoryAuditStore]:
    store = InMemoryPSGStore()
    audit_store = InMemoryAuditStore()
    consent_provider = StaticConsentProvider()
    if consent is not None:
        consent_provider.grant(patient_id, consent)
    profile_provider = StaticProfileProvider()
    if seed_profile:
        profile_provider.put(profile(patient_id))
    engine = PatientStateEngine(
        store=store,
        consent_provider=consent_provider,
        audit_writer=AuditWriter(audit_store),
        versions=VERSIONS,
        profile_provider=profile_provider,
        clock=lambda: OCCURRED_AT,
    )
    return engine, store, audit_store
