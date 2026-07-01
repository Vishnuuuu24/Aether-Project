"""Patient profile contract. Source of truth: docs/04 §1.

The first contract in the spec and load-bearing for baseline cold-start: a
population-range fallback cannot run until `sex_at_birth` and an age basis
(`dob` or `age_years`) are known (docs/04 §1, docs/05). The consent block is
embedded here; the consent *decision* logic lives in `core.auth`, not here.

`PatientProfile` and `SexAtBirth` previously lived in `schemas/consent.py`; they
were relocated here so the patient contract has a single dedicated home and
`consent.py` owns only the consent shape (CLAUDE.md: contracts defined once).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from .consent import Consent


class SexAtBirth(str, Enum):
    MALE = "male"
    FEMALE = "female"
    INTERSEX = "intersex"
    UNKNOWN = "unknown"


class PatientProfile(BaseModel):
    """Onboarding record (set once; later changes are versioned profile updates)."""

    patient_id: UUID = Field(default_factory=uuid4)
    consent: Consent
    dob: date | None = None
    age_years: int | None = Field(default=None, ge=0, le=150)
    sex_at_birth: SexAtBirth
    gender: str | None = None
    height_cm: float | None = Field(default=None, gt=0)
    weight_kg: float | None = Field(default=None, gt=0)
    weight_measured_at: datetime | None = None
    blood_group: str | None = None
    physical_disability: str | None = None

    @field_validator("weight_measured_at")
    @classmethod
    def _measured_at_tz(cls, v: datetime | None) -> datetime | None:
        # All timestamps are RFC 3339 with timezone (docs/04 header).
        if v is not None and v.tzinfo is None:
            raise ValueError("weight_measured_at must include timezone — naive datetimes rejected")
        return v

    @property
    def has_age_basis(self) -> bool:
        """An age basis exists (either dob or an explicit age) for fallback ranges."""
        return self.dob is not None or self.age_years is not None

    @property
    def population_fallback_ready(self) -> bool:
        """True when the load-bearing fields for a population-range fallback are
        populated (docs/04 §1). Whether intersex/unknown sex maps to a usable
        range is a clinical-input decision made downstream by the baseline engine;
        this only checks that the required inputs are present.
        """
        return self.has_age_basis and self.sex_at_birth is not SexAtBirth.UNKNOWN
