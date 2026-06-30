"""Per-reading schema. Source of truth: docs/04_Data_Contracts_and_Schemas.md §2.

Every single vitals data point in the system must conform to this model. No
service may redefine this contract locally — import it from here.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class MeasurementContext(str, Enum):
    RESTING = "resting"
    ACTIVE = "active"
    ASLEEP = "asleep"
    POST_MEAL = "post_meal"
    UNKNOWN = "unknown"


class MetricCode(str, Enum):
    """v1 supported metrics. Required core must work alone (docs/04 §2.1)."""

    HEART_RATE = "heart_rate"
    STEPS = "steps"
    SLEEP = "sleep"
    SPO2 = "spo2"
    RESPIRATORY_RATE = "respiratory_rate"
    SKIN_TEMP = "skin_temp"
    ECG = "ecg"
    HRV = "hrv"
    GLUCOSE = "glucose"
    BP = "bp"
    VO2MAX = "vo2max"
    GAIT = "gait"
    WEIGHT = "weight"
    MENSTRUAL_CYCLE = "menstrual_cycle"


REQUIRED_CORE_METRICS = {MetricCode.HEART_RATE, MetricCode.STEPS}


class Reading(BaseModel):
    """A single normalised vitals reading. See docs/04 §2 for the contract."""

    reading_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID
    metric_code: MetricCode
    value: float | dict[str, Any]
    unit: str
    timestamp: datetime
    source_device: str
    sqi: float = Field(ge=0.0, le=1.0)
    context: MeasurementContext
    included_in_baseline: bool = False
    ingest_adapter: str
    raw_ref: str | None = None

    @field_validator("timestamp")
    @classmethod
    def must_have_timezone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamp must include timezone — naive datetimes are rejected")
        return v

    @field_validator("unit")
    @classmethod
    def unit_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("unit is required and must be explicit, never assumed")
        return v


class IngestRejection(BaseModel):
    """Structured 422 error body for a rejected reading."""

    field: str
    issue: str


class IngestBatchResult(BaseModel):
    accepted: list[UUID]
    rejected: list[dict[str, Any]]  # {"index": int, "errors": list[IngestRejection]}
