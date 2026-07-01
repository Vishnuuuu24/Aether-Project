"""Patient profile validation + round-trip (docs/04 §1)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from schemas import Consent, ConsentScope, PatientProfile, SexAtBirth


def mk_consent() -> Consent:
    return Consent(
        scope=[ConsentScope.VITALS], version="v1", granted_at=datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_minimal_valid_profile() -> None:
    p = PatientProfile(consent=mk_consent(), sex_at_birth=SexAtBirth.FEMALE)
    assert p.patient_id is not None
    assert not p.has_age_basis
    assert not p.population_fallback_ready


def test_sex_at_birth_is_required() -> None:
    with pytest.raises(ValidationError):
        PatientProfile(consent=mk_consent())  # type: ignore[call-arg]


def test_population_fallback_ready_needs_age_and_known_sex() -> None:
    ready = PatientProfile(consent=mk_consent(), sex_at_birth=SexAtBirth.MALE, age_years=40)
    assert ready.has_age_basis and ready.population_fallback_ready

    with_dob = PatientProfile(
        consent=mk_consent(), sex_at_birth=SexAtBirth.FEMALE, dob=date(1990, 5, 1)
    )
    assert with_dob.population_fallback_ready


def test_population_fallback_blocked_when_sex_unknown() -> None:
    p = PatientProfile(consent=mk_consent(), sex_at_birth=SexAtBirth.UNKNOWN, age_years=40)
    assert p.has_age_basis
    assert not p.population_fallback_ready


def test_negative_age_and_nonpositive_measurements_rejected() -> None:
    with pytest.raises(ValidationError):
        PatientProfile(consent=mk_consent(), sex_at_birth=SexAtBirth.MALE, age_years=-1)
    with pytest.raises(ValidationError):
        PatientProfile(consent=mk_consent(), sex_at_birth=SexAtBirth.MALE, height_cm=0)
    with pytest.raises(ValidationError):
        PatientProfile(consent=mk_consent(), sex_at_birth=SexAtBirth.MALE, weight_kg=-5)


def test_naive_weight_timestamp_rejected() -> None:
    with pytest.raises(ValidationError):
        PatientProfile(
            consent=mk_consent(),
            sex_at_birth=SexAtBirth.MALE,
            weight_measured_at=datetime(2026, 6, 1),  # no tzinfo
        )


def test_roundtrip_serialise_validate() -> None:
    p = PatientProfile(
        consent=mk_consent(),
        sex_at_birth=SexAtBirth.FEMALE,
        age_years=33,
        height_cm=170.5,
        weight_kg=65.0,
        weight_measured_at=datetime(2026, 6, 1, 8, 0, tzinfo=UTC),
        blood_group="O+",
    )
    restored = PatientProfile.model_validate_json(p.model_dump_json())
    assert restored == p
