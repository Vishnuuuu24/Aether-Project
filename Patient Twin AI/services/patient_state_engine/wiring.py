"""DB-backed wiring for the Patient State Engine (docs/08; docs/15 T7.2).

The SAME engine, config-switched: the SqlAlchemy PSG + audit stores and DB-backed
consent/profile providers plug in behind the existing interfaces. No engine code
changes and no forked "DB service" — where things run is configuration, not code
(CLAUDE.md). The caller owns the transaction boundary (one transaction per request),
so a PSG node write and its audit record commit atomically.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.audit import AuditWriter
from core.audit.sql_store import SqlAlchemyAuditStore
from core.versioning import VersionSet
from schemas.consent import Consent, ConsentScope
from schemas.patient import PatientProfile, SexAtBirth

from .service import PatientStateEngine
from .sql_store import SqlAlchemyPSGStore

# A stand-in consent for a profile that has no consent row yet: empty scope means the
# consent gate denies by default (build_projection → 403), which is the correct outcome.
_NO_CONSENT = Consent(scope=[], version="none", granted_at=datetime(1970, 1, 1, tzinfo=UTC))


class SqlConsentProvider:
    """Reads the patient's current consent from the append-only `consent` table
    (latest row wins; a revoked row is still returned so the gate can see the flag).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_consent(self, patient_id: UUID) -> Consent | None:
        from core.db.models import Consent as ConsentRow

        row = self._session.execute(
            select(ConsentRow)
            .where(ConsentRow.patient_id == patient_id)
            .order_by(ConsentRow.created_at.desc(), ConsentRow.consent_id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if row is None:
            return None
        return Consent(
            scope=[ConsentScope(s) for s in row.scope],
            version=row.version,
            granted_at=row.granted_at,
            revoked_at=row.revoked_at,
        )


class SqlProfileProvider:
    """Reads the demographic profile from `patient_profile`. The projection only uses
    age/sex; the embedded consent is filled from the consent table for completeness.
    """

    def __init__(self, session: Session, consent_provider: SqlConsentProvider) -> None:
        self._session = session
        self._consent = consent_provider

    def get_profile(self, patient_id: UUID) -> PatientProfile | None:
        from core.db.models import PatientProfile as ProfileRow

        row = self._session.get(ProfileRow, patient_id)
        if row is None:
            return None
        return PatientProfile(
            patient_id=row.patient_id,
            consent=self._consent.get_consent(patient_id) or _NO_CONSENT,
            age_years=row.age_years,
            dob=row.dob,
            sex_at_birth=SexAtBirth(row.sex_at_birth),
        )


def build_sql_engine(session: Session, versions: VersionSet) -> PatientStateEngine:
    """A PatientStateEngine bound to Postgres via `session`. The PSG store, audit
    store, and consent/profile providers all share the one session/transaction.
    """
    consent = SqlConsentProvider(session)
    return PatientStateEngine(
        store=SqlAlchemyPSGStore(session),
        consent_provider=consent,
        audit_writer=AuditWriter(SqlAlchemyAuditStore(session)),
        versions=versions,
        profile_provider=SqlProfileProvider(session, consent),
    )
