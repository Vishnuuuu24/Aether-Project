"""Patient-profile lookup for the projection (age/sex demographics; docs/04 §1, §5).

The projection carries `patient_age_years` / `patient_sex_at_birth` for the LLM.
`StaticProfileProvider` is dev/test wiring; production reads the profile table.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from schemas.patient import PatientProfile


class ProfileProvider(Protocol):
    def get_profile(self, patient_id: UUID) -> PatientProfile | None: ...


class StaticProfileProvider:
    def __init__(self) -> None:
        self._by_patient: dict[UUID, PatientProfile] = {}

    def put(self, profile: PatientProfile) -> None:
        self._by_patient[profile.patient_id] = profile

    def get_profile(self, patient_id: UUID) -> PatientProfile | None:
        return self._by_patient.get(patient_id)
