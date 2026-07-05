"""DB-backed persistence + restart durability (docs/15 T7.2).

Drives the SAME PatientStateEngine against Postgres via `build_sql_engine`, commits a
baseline + deviation (and their audit records) in one transaction, then DISPOSES the
engine (a process restart) and re-opens a fresh connection to prove:
  * the PSG nodes are still there — `build_projection` reads them back;
  * the hash-chained audit trail persisted and `verify_chain` still holds.

Skips cleanly when Postgres is unreachable (scratch_db_url fixture).
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from core.audit import verify_chain
from core.db import make_session_factory
from core.db.base import Base
from core.db.models import AuditLog
from schemas.audit import AuditAction, AuditActor, AuditRecord
from services.patient_state_engine.wiring import build_sql_engine

from ._factories import VERSIONS, baseline, deviation

OCCURRED_AT = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)


def _to_record(row: AuditLog) -> AuditRecord:
    return AuditRecord(
        audit_id=row.audit_id,
        patient_id=row.patient_id,
        actor=AuditActor(row.actor),
        action=AuditAction(row.action),
        input_refs=row.input_refs,
        output_refs=row.output_refs,
        versions=row.versions,
        timestamp=row.timestamp,
        prev_hash=row.prev_hash,
        hash=row.hash,
    )


def _seed_patient(session_factory: sessionmaker[Session], pid: UUID) -> None:
    from core.db.models import Consent as ConsentRow
    from core.db.models import PatientProfile as ProfileRow

    with session_factory() as session:
        session.add(ProfileRow(patient_id=pid, sex_at_birth="female", age_years=41))
        session.flush()  # satisfy the consent → patient_profile FK before inserting consent
        session.add(
            ConsentRow(
                patient_id=pid,
                scope=["vitals"],
                version="v1",
                granted_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        session.commit()


def test_psg_and_audit_persist_across_restart(scratch_db_url: str) -> None:
    from core.db.models import AuditLog, BaselineNode

    engine = create_engine(scratch_db_url)
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    pid = uuid4()
    _seed_patient(factory, pid)

    # --- Write PSG + audit through the DB-backed engine, one transaction ---
    with factory() as session:
        state = build_sql_engine(session, VERSIONS)
        commit = state.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)
        assert commit.baseline_committed
        assert commit.deviation_node is not None
        session.commit()

    engine.dispose()  # ---- process restart: drop every connection ----

    # --- Re-open a fresh engine/session and read everything back ---
    engine2 = create_engine(scratch_db_url)
    factory2 = make_session_factory(engine2)
    try:
        with factory2() as session:
            # PSG node persisted through the restart.
            baselines = (
                session.execute(select(BaselineNode).where(BaselineNode.patient_id == pid))
                .scalars()
                .all()
            )
            assert len(baselines) == 1

            # The consent-scoped projection reads the persisted state back.
            projection = build_sql_engine(session, VERSIONS).build_projection(pid)
            assert projection.baselines
            assert projection.recent_deviations

            # Audit chain survived and still verifies end to end.
            rows = session.execute(select(AuditLog).order_by(AuditLog.seq)).scalars().all()
            records = [_to_record(r) for r in rows]
            assert len(records) >= 2  # baseline_update + state_commit (deviation)
            verify_chain(records)  # raises AuditChainError if tampered/broken
            assert {r.action for r in records} >= {
                AuditAction.BASELINE_UPDATE,
                AuditAction.STATE_COMMIT,
            }
    finally:
        engine2.dispose()
