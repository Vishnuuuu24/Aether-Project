"""SqlAlchemyPSGStore — the relational PSG persists and reads back, sharing one
transaction with the audit chain (append-only + audited, over real Postgres).
Skips when Postgres is unavailable.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import create_engine, select

from core.audit import AuditWriter, verify_chain
from core.audit.sql_store import SqlAlchemyAuditStore
from core.db import make_session_factory
from core.db import models as _models  # noqa: F401 -- register ORM tables before create_all
from core.db.base import Base
from schemas.audit import AuditAction, AuditActor, AuditRecord
from services.patient_state_engine.consent import StaticConsentProvider
from services.patient_state_engine.service import PatientStateEngine
from services.patient_state_engine.sql_store import SqlAlchemyPSGStore

from ._factories import (
    OCCURRED_AT,
    VERSIONS,
    baseline,
    deviation,
    event_candidate,
    forecast,
    forecast_consent,
    vitals_consent,
)


def _to_record(row: object) -> AuditRecord:
    r = row  # ORM AuditLog
    return AuditRecord(
        audit_id=r.audit_id,  # type: ignore[attr-defined]
        patient_id=r.patient_id,  # type: ignore[attr-defined]
        actor=AuditActor(r.actor),  # type: ignore[attr-defined]
        action=AuditAction(r.action),  # type: ignore[attr-defined]
        input_refs=r.input_refs,  # type: ignore[attr-defined]
        output_refs=r.output_refs,  # type: ignore[attr-defined]
        versions=r.versions,  # type: ignore[attr-defined]
        timestamp=r.timestamp,  # type: ignore[attr-defined]
        prev_hash=r.prev_hash,  # type: ignore[attr-defined]
        hash=r.hash,  # type: ignore[attr-defined]
    )


def test_relational_psg_persists_and_reads_back(scratch_db_url: str) -> None:
    engine = create_engine(scratch_db_url)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)
    pid = uuid4()

    consent_provider = StaticConsentProvider()
    consent_provider.grant(pid, vitals_consent())

    # Commit within one transaction: PSG node + audit record are atomic.
    with session_factory() as session:
        state = PatientStateEngine(
            store=SqlAlchemyPSGStore(session),
            consent_provider=consent_provider,
            audit_writer=AuditWriter(SqlAlchemyAuditStore(session)),
            versions=VERSIONS,
            clock=lambda: OCCURRED_AT,
        )
        result = state.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)
        assert result.baseline_node is not None and result.deviation_node is not None
        session.commit()

    # Fresh session: the relational PSG returns the persisted current state.
    with session_factory() as session:
        store = SqlAlchemyPSGStore(session)
        baselines = store.current_baselines(pid)
        deviations = store.recent_deviations(pid, limit=10)
    assert len(baselines) == 1
    assert baselines[0].center == 60.0
    assert baselines[0].version == 1
    assert len(deviations) == 1
    assert deviations[0].baseline_id == baselines[0].id

    # The audit chain persisted through Postgres verifies intact.
    from core.db.models import AuditLog

    with session_factory() as session:
        rows = session.execute(select(AuditLog).order_by(AuditLog.seq)).scalars().all()
    records = [_to_record(r) for r in rows]
    engine.dispose()
    assert len(records) == 2  # baseline_update + state_commit
    verify_chain(records)


def test_event_persists_and_reads_back(scratch_db_url: str) -> None:
    engine = create_engine(scratch_db_url)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)
    pid = uuid4()
    consent_provider = StaticConsentProvider()
    consent_provider.grant(pid, vitals_consent())

    with session_factory() as session:
        state = PatientStateEngine(
            store=SqlAlchemyPSGStore(session),
            consent_provider=consent_provider,
            audit_writer=AuditWriter(SqlAlchemyAuditStore(session)),
            versions=VERSIONS,
            clock=lambda: OCCURRED_AT,
        )
        node = state.commit_event(event_candidate(pid))
        session.commit()

    with session_factory() as session:
        events = SqlAlchemyPSGStore(session).active_events(pid)
    engine.dispose()

    assert len(events) == 1
    assert events[0].id == node.id
    assert events[0].status == "active"
    assert len(events[0].contributing_deviation_ids) == 2


def test_forecast_persists_and_reads_back(scratch_db_url: str) -> None:
    engine = create_engine(scratch_db_url)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)
    pid = uuid4()
    consent_provider = StaticConsentProvider()
    consent_provider.grant(pid, forecast_consent())

    with session_factory() as session:
        state = PatientStateEngine(
            store=SqlAlchemyPSGStore(session),
            consent_provider=consent_provider,
            audit_writer=AuditWriter(SqlAlchemyAuditStore(session)),
            versions=VERSIONS,
            clock=lambda: OCCURRED_AT,
        )
        node = state.commit_forecast(forecast(pid))
        session.commit()

    with session_factory() as session:
        forecasts = SqlAlchemyPSGStore(session).latest_forecasts(pid)
    engine.dispose()

    assert len(forecasts) == 1
    assert forecasts[0].id == node.id
    assert forecasts[0].points == [60.0, 61.0, 62.0]
    # intervals survive the JSONB round-trip as (lower, upper) tuples.
    assert forecasts[0].intervals[0] == (58.0, 62.0)


def test_new_version_supersedes_prior_in_db(scratch_db_url: str) -> None:
    engine = create_engine(scratch_db_url)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)
    pid = uuid4()
    consent_provider = StaticConsentProvider()
    consent_provider.grant(pid, vitals_consent())

    with session_factory() as session:
        state = PatientStateEngine(
            store=SqlAlchemyPSGStore(session),
            consent_provider=consent_provider,
            audit_writer=AuditWriter(SqlAlchemyAuditStore(session)),
            versions=VERSIONS,
            clock=lambda: OCCURRED_AT,
        )
        state.commit_deviation(baseline(pid, center=60.0), deviation(pid), occurred_at=OCCURRED_AT)
        state.commit_deviation(baseline(pid, center=72.0), deviation(pid), occurred_at=OCCURRED_AT)
        session.commit()

    from core.db.models import BaselineNode as BaselineRow

    with session_factory() as session:
        rows = (
            session.execute(select(BaselineRow).where(BaselineRow.patient_id == pid))
            .scalars()
            .all()
        )
        current = SqlAlchemyPSGStore(session).current_baseline(pid, "heart_rate", "resting")
    engine.dispose()

    assert len(rows) == 2  # both versions retained (append-only)
    assert current is not None and current.version == 2 and current.center == 72.0
    assert current.supersedes is not None
