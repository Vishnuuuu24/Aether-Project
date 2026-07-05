"""Every audit producer persists to the ONE global chain, verifiable across a restart
(docs/15 T7.2b; closes the gap where only the state engine was DB-backed).

Governance (consent + outcome), the copilot (output decision), and ingestion all
write through their app getters bound to a Postgres session; after disposing the
engine (a restart), a fresh connection reads the whole chain back and `verify_chain`
holds. Also asserts the getters fall back to the in-memory singletons with no session.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import core.db.models  # noqa: F401 — registers every table on Base.metadata for create_all
from core.audit import AuditWriter, verify_chain
from core.audit.sql_store import SqlAlchemyAuditStore
from core.db import make_session_factory
from core.db.base import Base
from schemas.audit import AuditAction
from schemas.consent import ConsentScope
from schemas.outcome import Outcome, OutcomeSource, OutcomeType
from schemas.output_contract import (
    Abstention,
    Escalation,
    Evidence,
    EvidenceKind,
    OutputContract,
    OutputType,
    PolicyDecision,
    PolicyRecord,
    RecommendedAction,
    Severity,
    VersionStamp,
)
from schemas.reading import MeasurementContext, MetricCode
from services.copilot_service.app import main as copilot_app
from services.copilot_service.audit import AuditWriterSink
from services.governance_service.app import main as governance_app
from services.ingestion_service.app import main as ingestion_app

_VERSIONS = VersionStamp(model="m1", ruleset="r1", baseline_engine="b1", prompt="p1")


def _output(pid: UUID) -> OutputContract:
    return OutputContract(
        patient_id=pid,
        type=OutputType.INFO,
        message="Your resting heart rate rose above your usual range.",
        severity=Severity.LOW,
        confidence=0.7,
        evidence=[Evidence(kind=EvidenceKind.KB_PASSAGE, ref="kb:1", quote_or_fact="RHR rises.")],
        recommended_action=RecommendedAction.MONITOR,
        escalation=Escalation(),
        abstained=Abstention(),
        policy=PolicyRecord(decision=PolicyDecision.APPROVED, rule_ids=["R2_grounding"]),
        versions=_VERSIONS,
        created_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )


def _outcome(pid: UUID) -> Outcome:
    now = datetime(2026, 6, 2, 9, 0, tzinfo=UTC)
    return Outcome(
        patient_id=pid,
        outcome_type=OutcomeType.DIAGNOSIS,
        occurred_at=now,
        detail="clinician-confirmed diagnosis",
        source=OutcomeSource.CLINICIAN,
        recorded_at=now,
    )


def _reading(pid: UUID) -> dict[str, object]:
    return {
        "patient_id": str(pid),
        "metric_code": MetricCode.HEART_RATE.value,
        "value": 62.0,
        "unit": "bpm",
        "timestamp": "2026-06-01T08:00:00+00:00",
        "source_device": "apple_watch_s9",
        "sqi": 0.9,
        "context": MeasurementContext.RESTING.value,
        "ingest_adapter": "csv",
    }


def _seed_profile(factory: sessionmaker[Session], pid: UUID) -> None:
    # Only the profile is seeded — consent comes from the governance grant below, so
    # the test also proves the write-through: a grant becomes visible to the consent
    # provider ingestion reads.
    from core.db.models import PatientProfile as ProfileRow

    with factory() as session:
        session.add(ProfileRow(patient_id=pid, sex_at_birth="female", age_years=41))
        session.commit()


def test_getters_fall_back_to_memory_singletons_without_a_session() -> None:
    # No DB session → each service uses its in-memory dev store (unchanged default).
    assert governance_app.get_consent_ledger(None) is governance_app._consent_ledger
    assert governance_app.get_outcome_store(None) is governance_app._outcome_store
    assert governance_app.get_audit_store(None) is governance_app._audit_store
    assert copilot_app.get_copilot(None) is copilot_app._copilot
    assert copilot_app.get_state_engine(None) is copilot_app._state_engine
    assert ingestion_app.get_service(None) is ingestion_app._service


def test_all_producers_share_one_persistent_verifiable_chain(scratch_db_url: str) -> None:
    from core.db.models import OutputRecord
    from services.copilot_service.output_store import SqlOutputStore
    from services.governance_service.sql_stores import SqlOutcomeRepo
    from services.patient_state_engine.wiring import SqlConsentProvider

    engine = create_engine(scratch_db_url)
    Base.metadata.create_all(engine)
    factory = make_session_factory(engine)
    pid = uuid4()
    _seed_profile(factory, pid)
    outcome, output = _outcome(pid), _output(pid)

    # Each producer writes through its APP GETTER bound to the Postgres session.
    with factory() as session:
        governance_app.get_consent_ledger(session).grant(
            pid, scope=[ConsentScope.VITALS], version="grant-v2"
        )  # CONSENT_CHANGE + a consent ROW
        governance_app.get_outcome_store(session).record(outcome)  # OUTCOME_CAPTURE + row
        SqlOutputStore(session).save(output)  # an output ROW
        AuditWriterSink(AuditWriter(SqlAlchemyAuditStore(session))).record(
            output
        )  # POLICY_DECISION (the copilot's audit sink)
        # Consent granted just above is visible to ingestion in the same transaction.
        result = ingestion_app.get_service(session).ingest([_reading(pid)], adapter="csv")  # INGEST
        assert len(result.accepted) == 1
        session.commit()

    engine.dispose()  # ---- restart ----

    engine2 = create_engine(scratch_db_url)
    try:
        with make_session_factory(engine2)() as session:
            # 1) The one hash chain survived and still verifies across every producer.
            records = SqlAlchemyAuditStore(session).records
            verify_chain(records)
            assert {
                AuditAction.CONSENT_CHANGE,
                AuditAction.OUTCOME_CAPTURE,
                AuditAction.POLICY_DECISION,
                AuditAction.INGEST,
            } <= {r.action for r in records}

            # 2) The ROWS persisted too — governance grant → consent provider sees it.
            consent = SqlConsentProvider(session).get_consent(pid)
            assert consent is not None and consent.version == "grant-v2"
            assert SqlOutcomeRepo(session).get(outcome.outcome_id) is not None
            assert session.get(OutputRecord, output.output_id) is not None
    finally:
        engine2.dispose()
