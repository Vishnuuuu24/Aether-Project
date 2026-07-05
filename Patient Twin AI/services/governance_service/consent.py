"""Consent lifecycle (grant / update / revoke), audit-backed. docs/04 §1, docs/07 §2.

The other services carry only a read-only `ConsentProvider` port; this ledger is
the production owner of the consent record. Every mutation mints a
`CONSENT_CHANGE` audit record into the same hash-chain, so a revocation can never
be silently lost. Deny-by-default: an unknown patient has no consent.

Consent is patient-driven, so mutations are audited with actor=PATIENT.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from core.audit import AuditWriter
from schemas.audit import AuditAction, AuditActor
from schemas.consent import Consent, ConsentScope


class ConsentStore(Protocol):
    """Where the ledger keeps the consent record. In-memory for dev, the append-only
    `consent` table in production (`SqlConsentStore`) — the store is the seam."""

    def current(self, patient_id: UUID) -> Consent | None: ...
    def append(self, patient_id: UUID, consent: Consent) -> None: ...


class InMemoryConsentStore:
    def __init__(self) -> None:
        self._by_patient: dict[UUID, Consent] = {}

    def current(self, patient_id: UUID) -> Consent | None:
        return self._by_patient.get(patient_id)

    def append(self, patient_id: UUID, consent: Consent) -> None:
        self._by_patient[patient_id] = consent


class ConsentLedger:
    def __init__(self, audit_writer: AuditWriter, store: ConsentStore | None = None) -> None:
        self._audit_writer = audit_writer
        self._store = store or InMemoryConsentStore()

    def get_consent(self, patient_id: UUID) -> Consent | None:
        """Current consent record (structurally satisfies the `ConsentProvider`
        port the other services depend on)."""
        return self._store.current(patient_id)

    def grant(
        self,
        patient_id: UUID,
        *,
        scope: list[ConsentScope],
        version: str,
        now: datetime | None = None,
    ) -> Consent:
        """Grant or update scoped consent. Replaces any prior record and audits it."""
        ts = now or datetime.now(UTC)
        consent = Consent(scope=scope, version=version, granted_at=ts)
        self._store.append(patient_id, consent)
        self._audit_writer.write(
            patient_id=patient_id,
            actor=AuditActor.PATIENT,
            action=AuditAction.CONSENT_CHANGE,
            output_refs=[f"consent:grant:{version}"],
            versions={"consent_version": version},
            timestamp=ts,
        )
        return consent

    def revoke(self, patient_id: UUID, *, now: datetime | None = None) -> Consent | None:
        """Revoke consent — stops all processing in scope (docs/07 §2). Idempotent:
        revoking an unknown or already-revoked record is a no-op that returns the
        current state without minting a spurious audit record."""
        current = self._store.current(patient_id)
        if current is None or current.revoked_at is not None:
            return current
        ts = now or datetime.now(UTC)
        revoked = current.model_copy(update={"revoked_at": ts})
        self._store.append(patient_id, revoked)
        self._audit_writer.write(
            patient_id=patient_id,
            actor=AuditActor.PATIENT,
            action=AuditAction.CONSENT_CHANGE,
            output_refs=[f"consent:revoke:{current.version}"],
            versions={"consent_version": current.version},
            timestamp=ts,
        )
        return revoked
