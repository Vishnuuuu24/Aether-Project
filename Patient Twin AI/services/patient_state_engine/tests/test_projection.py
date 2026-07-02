"""T1.4 DoD (read side): consent-scoped projection, no raw signals (docs/04 §5)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from core.audit import AuditWriter, InMemoryAuditStore
from core.auth.errors import ConsentError
from schemas.consent import Consent, ConsentScope
from services.patient_state_engine.consent import StaticConsentProvider
from services.patient_state_engine.profile import StaticProfileProvider
from services.patient_state_engine.service import PatientStateEngine, ProfileNotFoundError
from services.patient_state_engine.store import InMemoryPSGStore

from ._factories import (
    OCCURRED_AT,
    VERSIONS,
    baseline,
    deviation,
    profile,
    vitals_consent,
)


def _engine(
    pid: UUID, consent: Consent | None, *, seed_profile: bool = True
) -> tuple[PatientStateEngine, InMemoryPSGStore]:
    store = InMemoryPSGStore()
    consent_provider = StaticConsentProvider()
    if consent is not None:
        consent_provider.grant(pid, consent)
    profile_provider = StaticProfileProvider()
    if seed_profile:
        profile_provider.put(profile(pid))
    engine = PatientStateEngine(
        store=store,
        consent_provider=consent_provider,
        audit_writer=AuditWriter(InMemoryAuditStore()),
        versions=VERSIONS,
        profile_provider=profile_provider,
        clock=lambda: OCCURRED_AT,
    )
    return engine, store


def test_projection_includes_vitals_when_consented() -> None:
    pid = uuid4()
    engine, _ = _engine(pid, vitals_consent())
    engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)

    proj = engine.build_projection(pid)
    assert proj.consent_scope == ["vitals"]
    assert len(proj.baselines) == 1
    assert len(proj.recent_deviations) == 1
    assert proj.patient_age_years == 40
    assert proj.patient_sex_at_birth == "male"
    assert proj.as_of == OCCURRED_AT
    assert proj.versions.baseline_engine == "statistical-v1"


def test_projection_withholds_existing_vitals_when_scope_revoked() -> None:
    # Commit real VITALS state, then move consent to documents-only: the existing
    # baselines/deviations must be withheld — proving scoping actually gates data.
    pid = uuid4()
    store = InMemoryPSGStore()
    consent_provider = StaticConsentProvider()
    consent_provider.grant(pid, vitals_consent())
    profile_provider = StaticProfileProvider()
    profile_provider.put(profile(pid))
    engine = PatientStateEngine(
        store=store,
        consent_provider=consent_provider,
        audit_writer=AuditWriter(InMemoryAuditStore()),
        versions=VERSIONS,
        profile_provider=profile_provider,
        clock=lambda: OCCURRED_AT,
    )
    engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)
    assert len(engine.build_projection(pid).baselines) == 1  # present under vitals

    consent_provider.grant(
        pid,
        Consent(
            scope=[ConsentScope.DOCUMENTS],
            version="v2",
            granted_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )
    proj = engine.build_projection(pid)
    assert proj.consent_scope == ["documents"]
    assert proj.baselines == []  # withheld despite still existing in the store
    assert proj.recent_deviations == []


def test_projection_denied_without_any_consent() -> None:
    pid = uuid4()
    engine, _ = _engine(pid, None)
    with pytest.raises(ConsentError):
        engine.build_projection(pid)


def test_projection_profile_not_found() -> None:
    pid = uuid4()
    engine, _ = _engine(pid, vitals_consent(), seed_profile=False)
    with pytest.raises(ProfileNotFoundError):
        engine.build_projection(pid)


def test_projection_carries_no_raw_signals() -> None:
    pid = uuid4()
    engine, _ = _engine(pid, vitals_consent())
    engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)

    payload = engine.build_projection(pid).model_dump_json()
    # Summaries DO carry the modelled fields (proves data actually flowed through)...
    assert '"center"' in payload and '"z_robust"' in payload
    # ...but none of the node-level / raw fields ever leak into the projection.
    for forbidden in (
        "raw_ref",
        "reading_id",
        "sample_n",
        "window_spec",
        "method",
        "baseline_id",
        "supersedes",
    ):
        assert forbidden not in payload, f"projection leaked node-level field: {forbidden}"
