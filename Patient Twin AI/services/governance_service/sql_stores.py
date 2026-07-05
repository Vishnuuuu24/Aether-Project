"""Postgres adapters for the governance stores (docs/04 §1, §7; docs/15 T7.2c).

`SqlConsentStore` realises the append-only `consent` table (so a grant is visible to
every service's `SqlConsentProvider`); `SqlOutcomeRepo` realises the `outcome` table.
Both convert between the pydantic contracts and the ORM rows and do NOT commit — the
caller owns the transaction (one request, one atomic write + audit).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from schemas.consent import Consent, ConsentScope
from schemas.outcome import Outcome, OutcomeSource, OutcomeType


class SqlConsentStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def current(self, patient_id: UUID) -> Consent | None:
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

    def append(self, patient_id: UUID, consent: Consent) -> None:
        from core.db.models import Consent as ConsentRow

        self._session.add(
            ConsentRow(
                patient_id=patient_id,
                scope=[s.value for s in consent.scope],
                version=consent.version,
                granted_at=consent.granted_at,
                revoked_at=consent.revoked_at,
            )
        )
        self._session.flush()


class SqlOutcomeRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, outcome: Outcome) -> None:
        from core.db.models import Outcome as OutcomeRow

        self._session.add(
            OutcomeRow(
                outcome_id=outcome.outcome_id,
                patient_id=outcome.patient_id,
                outcome_type=outcome.outcome_type.value,
                occurred_at=outcome.occurred_at,
                detail=outcome.detail,
                code=outcome.code,
                linked_output_ids=[str(oid) for oid in outcome.linked_output_ids],
                versions=outcome.versions,
                source=outcome.source.value,
                recorded_at=outcome.recorded_at,
            )
        )
        self._session.flush()

    def get(self, outcome_id: UUID) -> Outcome | None:
        from core.db.models import Outcome as OutcomeRow

        row = self._session.get(OutcomeRow, outcome_id)
        return _to_schema(row) if row is not None else None

    def for_patient(self, patient_id: UUID) -> list[Outcome]:
        from core.db.models import Outcome as OutcomeRow

        rows = (
            self._session.execute(
                select(OutcomeRow)
                .where(OutcomeRow.patient_id == patient_id)
                .order_by(OutcomeRow.recorded_at.desc())
            )
            .scalars()
            .all()
        )
        return [_to_schema(row) for row in rows]


def _to_schema(row: Any) -> Outcome:
    return Outcome(
        outcome_id=row.outcome_id,
        patient_id=row.patient_id,
        outcome_type=OutcomeType(row.outcome_type),
        occurred_at=row.occurred_at,
        detail=row.detail,
        code=row.code,
        linked_output_ids=[UUID(x) for x in row.linked_output_ids],
        versions=dict(row.versions),
        source=OutcomeSource(row.source),
        recorded_at=row.recorded_at,
    )
