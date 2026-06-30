"""End-to-end audit chain over the real Postgres store: write → read back → verify,
and confirm a row tampered in the database fails verification.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from core.audit import AuditChainError, AuditWriter, verify_chain
from core.audit.sql_store import SqlAlchemyAuditStore
from core.db import make_session_factory
from core.db.base import Base
from core.db.models import AuditLog
from schemas.audit import AuditAction, AuditActor, AuditRecord


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


def _write_three(session_factory, patient_id) -> None:
    with session_factory() as session:
        writer = AuditWriter(SqlAlchemyAuditStore(session))
        for i in range(3):
            writer.write(
                patient_id=patient_id,
                actor=AuditActor.SYSTEM,
                action=AuditAction.INGEST,
                input_refs=[f"reading:{i}"],
                versions={"model": "m1"},
            )
        session.commit()


def test_sql_store_chain_roundtrip_verifies(scratch_db_url: str) -> None:
    engine = create_engine(scratch_db_url)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)
    pid = uuid4()

    _write_three(session_factory, pid)

    with session_factory() as session:
        rows = session.execute(select(AuditLog).order_by(AuditLog.seq)).scalars().all()
    records = [_to_record(r) for r in rows]
    engine.dispose()

    assert len(records) == 3
    assert records[0].prev_hash == "0" * 64
    verify_chain(records)  # intact chain persisted through Postgres → must not raise


def test_sql_store_db_tamper_detected(scratch_db_url: str) -> None:
    engine = create_engine(scratch_db_url)
    Base.metadata.create_all(engine)
    session_factory = make_session_factory(engine)
    pid = uuid4()
    _write_three(session_factory, pid)

    # Mutate a persisted row out from under the chain.
    with session_factory() as session:
        rows = session.execute(select(AuditLog).order_by(AuditLog.seq)).scalars().all()
        rows[1].input_refs = ["HACKED"]
        session.commit()

    with session_factory() as session:
        rows = session.execute(select(AuditLog).order_by(AuditLog.seq)).scalars().all()
        records = [_to_record(r) for r in rows]
    engine.dispose()

    with pytest.raises(AuditChainError) as exc:
        verify_chain(records)
    assert exc.value.index == 1
