"""Consent lookup for the state engine (deny-by-default; docs/02 §2).

Decision logic lives in `core.auth.consent_gate`; this only supplies the patient's
current consent record. Mirrors `services.ingestion_service.consent` — a service-
local port, not a contract. In v1 the governance service backs this in production.
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
