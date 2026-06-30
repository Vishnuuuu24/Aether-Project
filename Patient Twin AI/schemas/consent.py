"""Patient profile + consent schema. docs/04 §1.

No processing proceeds without a valid, scoped, non-revoked consent record
covering the relevant scope. This is enforced in core/auth, not here — this
module only defines the shape.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ConsentScope(str, Enum):
    VITALS = "vitals"
    DOCUMENTS = "documents"
    COPILOT = "copilot"
    FORECAST = "forecast"


class SexAtBirth(str, Enum):
    MALE = "male"
    FEMALE = "female"
    INTERSEX = "intersex"
    UNKNOWN = "unknown"


class Consent(BaseModel):
    scope: list[ConsentScope]
    version: str
    granted_at: datetime
    revoked_at: datetime | None = None

    def covers(self, required: ConsentScope) -> bool:
        return self.revoked_at is None and required in self.scope


class PatientProfile(BaseModel):
    patient_id: UUID = Field(default_factory=uuid4)
    consent: Consent
    dob: date | None = None
    age_years: int | None = None
    sex_at_birth: SexAtBirth
    gender: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    weight_measured_at: datetime | None = None
    blood_group: str | None = None
    physical_disability: str | None = None
