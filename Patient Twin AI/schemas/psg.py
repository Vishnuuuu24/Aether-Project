"""Patient State Graph (PSG): node types and the read-only projection handed
to the LLM. Source of truth: docs/02 §4 and docs/04 §3, §5.

Rules encoded here:
  - Every node is versioned + append-only (supersedes, never mutated).
  - PSGProjection is the ONLY thing the LLM may see. It carries no raw
    signal arrays and no `Reading.raw_ref`. Consent-scoped fields are
    omitted, not nulled, when out of scope.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .reading import MeasurementContext, MetricCode


class VersionedNode(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    patient_id: UUID
    version: int = 1
    supersedes: UUID | None = None
    created_at: datetime
    created_by: str  # service/actor name, for audit


class BaselineNode(VersionedNode):
    metric_code: MetricCode
    context: MeasurementContext
    method: str  # e.g. "robust_median_mad"
    center: float
    dispersion: float
    sample_n: int
    window_spec: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_population_fallback: bool


class DeviationDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"  # reading sits at the baseline center — no deviation


class DeviationNode(VersionedNode):
    metric_code: MetricCode
    baseline_id: UUID
    magnitude: float
    direction: DeviationDirection
    z_robust: float
    confidence: float = Field(ge=0.0, le=1.0)
    is_population_fallback: bool


class EventSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EventNode(VersionedNode):
    type: str
    severity: EventSeverity
    status: str  # "active" | "resolved"
    onset_ts: datetime
    contributing_deviation_ids: list[UUID]


class ConditionNode(VersionedNode):
    snomed_code: str
    display: str
    status: str
    onset: datetime | None = None
    source_document_id: UUID | None = None


class MedicationNode(VersionedNode):
    rxnorm_code: str
    display: str
    dose: str | None = None
    status: str
    source_document_id: UUID | None = None


class AllergyNode(VersionedNode):
    substance_code: str
    reaction: str
    severity: str
    source: str
    status: str = "committed"  # proposed | committed (docs/04 §4)


class ObservationNode(VersionedNode):
    loinc_code: str
    display: str
    value: str
    unit: str
    ts: datetime
    source_document_id: UUID | None = None
    status: str = "committed"  # proposed | committed (docs/04 §4)


class ForecastNode(VersionedNode):
    metric_code: MetricCode
    horizon_days: int
    points: list[float]
    intervals: list[tuple[float, float]]
    method: str
    generated_at: datetime


class DocumentNode(VersionedNode):
    """Provenance for a coded document (docs/02 §4.1, docs/04 §4). Coded clinical
    nodes reference it via `source_document_id`.
    """

    doc_type: str
    uri: str | None = None
    ocr_ref: str | None = None
    codes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Projection: the ONLY structure the LLM Gateway is allowed to receive.
# ---------------------------------------------------------------------------


class BaselineSummary(BaseModel):
    metric_code: MetricCode
    context: MeasurementContext
    center: float
    dispersion: float
    confidence: float
    is_population_fallback: bool


class DeviationSummary(BaseModel):
    metric_code: MetricCode
    direction: DeviationDirection
    magnitude: float
    z_robust: float
    confidence: float
    ts: datetime


class EventSummary(BaseModel):
    type: str
    severity: EventSeverity
    onset_ts: datetime


class ConditionSummary(BaseModel):
    snomed_code: str
    display: str
    status: str


class MedicationSummary(BaseModel):
    rxnorm_code: str
    display: str
    status: str


class AllergySummary(BaseModel):
    substance: str
    reaction: str
    severity: str


class ObservationSummary(BaseModel):
    loinc_code: str
    display: str
    value: str
    unit: str
    ts: datetime


class ForecastSummary(BaseModel):
    metric_code: MetricCode
    horizon_days: int
    points: list[float]
    intervals: list[tuple[float, float]]


class DocumentSummary(BaseModel):
    """A document *reference* + its coding result for the read API (docs/07 §4).

    Carries no OCR text and no raw signals — only the document type, the codes the
    coder emitted, and when it was committed. Per-code confirmation status lives on
    the coded entity nodes (surfaced via conditions/medications/observations).
    """

    doc_type: str
    codes: list[str]
    ts: datetime


class VersionStamp(BaseModel):
    baseline_engine: str
    ruleset: str
    prompt: str
    model: str


class PSGProjection(BaseModel):
    """Consent-scoped snapshot. No raw signals. No reading-level data.

    Built exclusively by the Patient State Engine. The LLM Gateway must
    refuse any call whose context is not an instance of this model.
    """

    patient_age_years: int | None
    patient_sex_at_birth: str
    baselines: list[BaselineSummary] = Field(default_factory=list)
    recent_deviations: list[DeviationSummary] = Field(default_factory=list)
    active_events: list[EventSummary] = Field(default_factory=list)
    conditions: list[ConditionSummary] = Field(default_factory=list)
    medications: list[MedicationSummary] = Field(default_factory=list)
    allergies: list[AllergySummary] = Field(default_factory=list)
    recent_observations: list[ObservationSummary] = Field(default_factory=list)
    latest_forecasts: list[ForecastSummary] = Field(default_factory=list)
    as_of: datetime
    consent_scope: list[str]
    versions: VersionStamp
