"""T0.1 DoD anchor — 'round-trip serialise/validate tests pass' (docs/04).

A single parametrised sweep that every data contract survives a JSON
serialise → deserialise → equality round-trip, plus the two headline rejection
rules the spec calls out for readings (missing timezone, missing unit). The
per-contract modules (test_reading, test_patient, …) cover field-level rules in
depth; this file is the consolidated serialisation guarantee.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError

from schemas import (
    AuditRecord,
    Consent,
    ConsentScope,
    Evidence,
    EvidenceKind,
    MeasurementContext,
    MetricCode,
    OutputContract,
    OutputType,
    PatientProfile,
    PolicyDecision,
    PolicyRecord,
    PSGProjection,
    Reading,
    RecommendedAction,
    Severity,
    SexAtBirth,
    VectorPayload,
    VectorSourceType,
)
from schemas.audit import AuditAction, AuditActor
from schemas.output_contract import VersionStamp as OutputVersionStamp
from schemas.psg import BaselineSummary
from schemas.psg import VersionStamp as ProjectionVersionStamp

TS = datetime(2026, 6, 1, 8, 0, tzinfo=UTC)


def a_consent() -> Consent:
    return Consent(scope=[ConsentScope.VITALS, ConsentScope.COPILOT], version="v1", granted_at=TS)


def a_patient() -> PatientProfile:
    return PatientProfile(
        consent=a_consent(),
        sex_at_birth=SexAtBirth.FEMALE,
        age_years=33,
        height_cm=170.0,
        weight_kg=64.0,
        weight_measured_at=TS,
        blood_group="O+",
    )


def a_reading() -> Reading:
    return Reading(
        patient_id=uuid4(),
        metric_code=MetricCode.HEART_RATE,
        value=58,
        unit="bpm",
        timestamp=TS,
        source_device="apple_watch_s9",
        sqi=0.95,
        context=MeasurementContext.RESTING,
        ingest_adapter="healthkit",
    )


def a_vector() -> VectorPayload:
    return VectorPayload(
        source_type=VectorSourceType.PATIENT_RECORD,
        patient_id=uuid4(),
        source_document_id=uuid4(),
        chunk_text="Discharge summary: stable, follow up in 2 weeks.",
        chunk_index=0,
        embedding_model="medcpt",
        consent_scope=ConsentScope.DOCUMENTS,
        codes=["LOINC:8867-4"],
        timestamp=TS,
    )


def a_projection() -> PSGProjection:
    return PSGProjection(
        patient_age_years=40,
        patient_sex_at_birth="male",
        baselines=[
            BaselineSummary(
                metric_code=MetricCode.HEART_RATE,
                context=MeasurementContext.RESTING,
                center=58.0,
                dispersion=4.0,
                confidence=0.9,
                is_population_fallback=False,
            )
        ],
        as_of=TS,
        consent_scope=["vitals"],
        versions=ProjectionVersionStamp(
            baseline_engine="b1", ruleset="r1", prompt="p1", model="m1"
        ),
    )


def an_output() -> OutputContract:
    return OutputContract(
        patient_id=uuid4(),
        type=OutputType.INFO,
        message="Your resting heart rate is within your normal range.",
        severity=Severity.NONE,
        confidence=0.9,
        evidence=[
            Evidence(
                kind=EvidenceKind.PSG_FACT,
                ref="baseline:heart_rate:resting",
                quote_or_fact="resting HR baseline centre 58 bpm",
            )
        ],
        recommended_action=RecommendedAction.NONE,
        policy=PolicyRecord(decision=PolicyDecision.APPROVED),
        versions=OutputVersionStamp(model="m1", ruleset="r1", baseline_engine="b1", prompt="p1"),
        created_at=TS,
    )


def an_audit_record() -> AuditRecord:
    return AuditRecord(
        patient_id=uuid4(),
        actor=AuditActor.SYSTEM,
        action=AuditAction.INGEST,
        input_refs=["reading:1"],
        versions={"model": "m1"},
        timestamp=TS,
        prev_hash="0" * 64,
        hash="a" * 64,
    )


CONTRACT_FACTORIES: list[Callable[[], BaseModel]] = [
    a_consent,
    a_patient,
    a_reading,
    a_vector,
    a_projection,
    an_output,
    an_audit_record,
]


@pytest.mark.parametrize("factory", CONTRACT_FACTORIES, ids=lambda f: f.__name__)
def test_contract_roundtrips_through_json(factory: Callable[[], BaseModel]) -> None:
    instance = factory()
    restored = type(instance).model_validate_json(instance.model_dump_json())
    assert restored == instance


def test_valid_reading_and_projection_parse_from_raw_json() -> None:
    # Valid JSON parses into the models (the plan's positive cases).
    assert Reading.model_validate_json(a_reading().model_dump_json()).unit == "bpm"
    assert (
        PSGProjection.model_validate_json(a_projection().model_dump_json()).patient_age_years == 40
    )


def _reading_kwargs(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "patient_id": uuid4(),
        "metric_code": MetricCode.HEART_RATE,
        "value": 58,
        "unit": "bpm",
        "timestamp": TS,
        "source_device": "apple_watch_s9",
        "sqi": 0.9,
        "context": MeasurementContext.RESTING,
        "ingest_adapter": "healthkit",
    }
    data.update(overrides)
    return data


def test_reading_without_timezone_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Reading(**_reading_kwargs(timestamp=datetime(2026, 6, 1, 7, 30)))  # naive


def test_reading_missing_unit_is_rejected() -> None:
    kwargs = _reading_kwargs()
    del kwargs["unit"]  # required field absent
    with pytest.raises(ValidationError):
        Reading(**kwargs)
