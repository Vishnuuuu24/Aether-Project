"""T1.4 DoD (commit side): state changes are append-only + audited; versioned PSG.
docs/04 §3, §7; docs/05 §8.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from core.audit import AuditWriter, InMemoryAuditStore, verify_chain
from core.auth.errors import ConsentError
from schemas.audit import AuditAction
from schemas.baseline import BaselineAvailability, DeviationMagnitude
from schemas.psg import DeviationDirection
from schemas.reading import MetricCode
from services.patient_state_engine.consent import StaticConsentProvider
from services.patient_state_engine.service import PatientStateEngine
from services.patient_state_engine.store import InMemoryPSGStore

from ._factories import (
    OCCURRED_AT,
    VERSIONS,
    baseline,
    deviation,
    event_candidate,
    fallback_baseline,
    forecast,
    forecast_consent,
    vitals_consent,
    wired_engine,
)


def test_commit_writes_baseline_and_deviation() -> None:
    pid = uuid4()
    engine, store, audit = wired_engine(pid, consent=vitals_consent())

    result = engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)

    assert result.baseline_node is not None and result.baseline_committed is True
    assert result.deviation_node is not None
    assert result.baseline_node.version == 1
    assert result.deviation_node.baseline_id == result.baseline_node.id
    # Deviation stores |z| as magnitude and the direction/z from the result.
    assert result.deviation_node.magnitude == pytest.approx(6.0)
    assert result.deviation_node.direction is DeviationDirection.UP
    assert len(store.current_baselines(pid)) == 1
    assert len(store.recent_deviations(pid, limit=10)) == 1

    # Every mutation audited, chain intact, versions stamped.
    actions = [r.action for r in audit.records]
    assert AuditAction.BASELINE_UPDATE in actions
    assert AuditAction.STATE_COMMIT in actions
    assert all(r.versions.get("baseline_engine") == "statistical-v1" for r in audit.records)
    verify_chain(audit.records)


def test_baseline_new_version_on_material_change() -> None:
    pid = uuid4()
    engine, store, _ = wired_engine(pid, consent=vitals_consent())

    engine.commit_deviation(baseline(pid, center=60.0), deviation(pid), occurred_at=OCCURRED_AT)
    result2 = engine.commit_deviation(
        baseline(pid, center=70.0), deviation(pid), occurred_at=OCCURRED_AT
    )

    assert result2.baseline_committed is True
    assert result2.baseline_node is not None
    assert result2.baseline_node.version == 2
    current = store.current_baseline(pid, "heart_rate", "resting")
    assert current is not None and current.version == 2 and current.center == 70.0
    assert current.supersedes is not None  # links back to v1 (append-only history)


def test_no_new_version_when_unchanged() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent())

    engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)
    again = engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)

    assert again.baseline_committed is False
    assert again.baseline_node is not None and again.baseline_node.version == 1


def test_normal_reading_commits_no_deviation() -> None:
    pid = uuid4()
    engine, store, _ = wired_engine(pid, consent=vitals_consent())

    result = engine.commit_deviation(
        baseline(pid),
        deviation(pid, magnitude=DeviationMagnitude.NORMAL, z_robust=0.5),
        occurred_at=OCCURRED_AT,
    )

    assert result.baseline_node is not None  # baseline still tracked
    assert result.deviation_node is None  # abstain: a normal reading is not a deviation
    assert store.recent_deviations(pid, limit=10) == []


def test_unavailable_baseline_commits_nothing() -> None:
    pid = uuid4()
    engine, store, audit = wired_engine(pid, consent=vitals_consent())

    result = engine.commit_deviation(
        baseline(
            pid,
            availability=BaselineAvailability.UNAVAILABLE,
            center=None,
            dispersion_sigma=None,
            is_population_fallback=False,
            sample_n=0,
        ),
        deviation(pid, availability=BaselineAvailability.UNAVAILABLE),
        occurred_at=OCCURRED_AT,
    )

    assert result.baseline_node is None
    assert result.deviation_node is None
    assert store.current_baselines(pid) == []
    assert audit.records == []  # nothing mutated => nothing audited


def test_fallback_to_personalised_transition_is_flagged_and_audited() -> None:
    pid = uuid4()
    engine, _, audit = wired_engine(pid, consent=vitals_consent())

    engine.commit_deviation(
        fallback_baseline(pid),
        deviation(
            pid,
            availability=BaselineAvailability.POPULATION_FALLBACK,
            is_population_fallback=True,
        ),
        occurred_at=OCCURRED_AT,
    )
    result = engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)

    assert result.transition is True
    assert result.baseline_node is not None and result.baseline_node.is_population_fallback is False
    # The transition is recorded in the audit trail, never silent (docs/05 §8).
    transition_refs = [ref for r in audit.records for ref in r.input_refs if "transition" in ref]
    assert transition_refs == ["transition:population_fallback->personalised"]


def test_commit_without_consent_denied() -> None:
    pid = uuid4()
    engine, store, audit = wired_engine(pid, consent=None)  # deny-by-default

    with pytest.raises(ConsentError):
        engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)
    assert store.current_baselines(pid) == []
    assert audit.records == []


def test_down_deviation_maps_direction_and_abs_magnitude() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent())

    result = engine.commit_deviation(
        baseline(pid),
        deviation(
            pid,
            direction=DeviationDirection.DOWN,
            z_robust=-6.0,
            magnitude=DeviationMagnitude.MARKED,
        ),
        occurred_at=OCCURRED_AT,
    )

    assert result.deviation_node is not None
    assert result.deviation_node.direction is DeviationDirection.DOWN
    assert result.deviation_node.z_robust == -6.0
    assert result.deviation_node.magnitude == pytest.approx(6.0)  # |z|, not the signed value


def test_commit_mismatched_metric_context_rejected() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent())
    with pytest.raises(ValueError, match="metric_code/context disagree"):
        engine.commit_deviation(
            baseline(pid, metric_code=MetricCode.HEART_RATE),
            deviation(pid, metric_code=MetricCode.SPO2),
            occurred_at=OCCURRED_AT,
        )


def test_store_scopes_by_patient_and_key() -> None:
    p1, p2 = uuid4(), uuid4()
    store = InMemoryPSGStore()
    consent_provider = StaticConsentProvider()
    consent_provider.grant(p1, vitals_consent())
    consent_provider.grant(p2, vitals_consent())
    engine = PatientStateEngine(
        store=store,
        consent_provider=consent_provider,
        audit_writer=AuditWriter(InMemoryAuditStore()),
        versions=VERSIONS,
        clock=lambda: OCCURRED_AT,
    )

    # p1: two metric series; p2: one.
    engine.commit_deviation(
        baseline(p1, metric_code=MetricCode.HEART_RATE),
        deviation(p1, metric_code=MetricCode.HEART_RATE),
        occurred_at=OCCURRED_AT,
    )
    engine.commit_deviation(
        baseline(p1, metric_code=MetricCode.SPO2),
        deviation(p1, metric_code=MetricCode.SPO2),
        occurred_at=OCCURRED_AT,
    )
    engine.commit_deviation(
        baseline(p2, metric_code=MetricCode.HEART_RATE),
        deviation(p2, metric_code=MetricCode.HEART_RATE),
        occurred_at=OCCURRED_AT,
    )

    assert len(store.current_baselines(p1)) == 2  # per-key: two distinct metrics
    assert len(store.current_baselines(p2)) == 1  # per-patient isolation
    assert len(store.recent_deviations(p1, limit=10)) == 2
    assert len(store.recent_deviations(p2, limit=10)) == 1


def test_commit_event_appends_and_audits() -> None:
    pid = uuid4()
    engine, store, audit = wired_engine(pid, consent=vitals_consent())

    node = engine.commit_event(event_candidate(pid))

    assert node.status == "active"
    assert node.created_by == "patient-state-engine"
    active = store.active_events(pid)
    assert len(active) == 1 and active[0].id == node.id
    # Audited as a PSG mutation, chain intact, event + rule referenced.
    assert AuditAction.STATE_COMMIT in [r.action for r in audit.records]
    refs = [ref for r in audit.records for ref in r.output_refs]
    assert f"event:{node.id}" in refs
    assert any("rule:stress-1" in ref for r in audit.records for ref in r.input_refs)
    verify_chain(audit.records)


def test_commit_event_without_consent_denied() -> None:
    pid = uuid4()
    engine, store, audit = wired_engine(pid, consent=None)
    with pytest.raises(ConsentError):
        engine.commit_event(event_candidate(pid))
    assert store.active_events(pid) == []
    assert audit.records == []


def test_commit_forecast_appends_and_audits() -> None:
    pid = uuid4()
    engine, store, audit = wired_engine(pid, consent=forecast_consent())

    node = engine.commit_forecast(forecast(pid))

    assert node.created_by == "patient-state-engine"
    assert node.horizon_days == 3
    forecasts = store.latest_forecasts(pid)
    assert len(forecasts) == 1 and forecasts[0].id == node.id
    refs = [ref for r in audit.records for ref in r.output_refs]
    assert f"forecast:{node.id}" in refs
    verify_chain(audit.records)


def test_commit_forecast_requires_forecast_consent() -> None:
    # VITALS-only consent must NOT be able to commit a forecast (separate scope).
    pid = uuid4()
    engine, store, _ = wired_engine(pid, consent=vitals_consent())
    with pytest.raises(ConsentError):
        engine.commit_forecast(forecast(pid))
    assert store.latest_forecasts(pid) == []


def test_naive_occurred_at_rejected() -> None:
    from datetime import datetime

    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent())
    with pytest.raises(ValueError, match="timezone-aware"):
        engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=datetime(2026, 6, 1))
