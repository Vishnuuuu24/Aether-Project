"""Consent lookup for ingestion (docs/02 §2 — ingestion performs the consent check).

The decision logic lives in `core.auth.consent_gate`; this just supplies the
patient's current consent record. In v1 the governance service will back this;
`StaticConsentProvider` is an in-memory, deny-by-default implementation used for
dev/tests (unknown patient → no consent → blocked).
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from schemas.consent import Consent


class ConsentProvider(Protocol):
    def get_consent(self, patient_id: UUID) -> Consent | None: ...


class StaticConsentProvider:
    """In-memory consent store. Deny-by-default: only seeded patients are covered."""

    def __init__(self) -> None:
        self._by_patient: dict[UUID, Consent] = {}

    def grant(self, patient_id: UUID, consent: Consent) -> None:
        self._by_patient[patient_id] = consent

    def get_consent(self, patient_id: UUID) -> Consent | None:
        return self._by_patient.get(patient_id)
