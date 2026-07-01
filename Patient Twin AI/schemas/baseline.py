"""Baseline-engine contracts (docs/02 §6, docs/05).

Interface-level types returned by `BaselineEngine`: `Baseline` (the per-`(patient,
metric, context)` estimate of *this user's normal*) and `DeviationResult` (a reading
scored against it). These are distinct from the persisted PSG nodes in `schemas.psg`
(`BaselineNode` / `DeviationNode`) — the Patient State Engine (T1.4) maps these
lightweight results into versioned, append-only nodes when it commits them.

CLAUDE.md: contracts live in schemas/ only; no service redefines them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from .psg import DeviationDirection
from .reading import MeasurementContext, MetricCode


class BaselineAvailability(str, Enum):
    """Where a baseline's numbers come from (docs/05 §4)."""

    PERSONALISED = "personalised"  # met sufficiency: this user's own normal
    POPULATION_FALLBACK = "population_fallback"  # labelled cold-start (docs/05 §4.1)
    UNAVAILABLE = "unavailable"  # no sufficiency and no population range configured


class DeviationMagnitude(str, Enum):
    """Bucketed |z_robust| (docs/05 §5). Thresholds are statistical, doc-specified."""

    NORMAL = "normal"  # |z| < 2
    MILD = "mild"  # 2 <= |z| < 3
    MODERATE = "moderate"  # 3 <= |z| < 4.5
    MARKED = "marked"  # |z| >= 4.5


class PopulationRange(BaseModel):
    """An age/sex population reference range — the labelled cold-start fallback
    (docs/05 §4.1). Values are clinical config, never fabricated by the engine.
    """

    low: float
    high: float
    unit: str

    @model_validator(mode="after")
    def _ordered(self) -> PopulationRange:
        if self.high < self.low:
            raise ValueError("population range high must be >= low")
        return self


class Baseline(BaseModel):
    """The current baseline for one `(patient, metric_code, context)` (docs/05 §4)."""

    patient_id: UUID
    metric_code: MetricCode
    context: MeasurementContext
    availability: BaselineAvailability
    center: float | None = None
    dispersion_sigma: float | None = None  # robust sigma = mad_scale * MAD
    ewma: float | None = None
    sample_n: int = Field(ge=0)
    span_days: float = Field(ge=0.0)
    window_days: int
    min_n: int
    min_days: int
    is_population_fallback: bool
    circadian_bucket: str | None = None  # set when the baseline is time-of-day stratified
    as_of: datetime | None = None
    method: str = "robust_median_mad"
    baseline_engine_version: str

    @model_validator(mode="after")
    def _consistency(self) -> Baseline:
        if self.is_population_fallback != (
            self.availability == BaselineAvailability.POPULATION_FALLBACK
        ):
            raise ValueError(
                "is_population_fallback must track availability == POPULATION_FALLBACK"
            )
        if self.availability == BaselineAvailability.PERSONALISED and (
            self.center is None or self.dispersion_sigma is None
        ):
            raise ValueError("a personalised baseline must have center and dispersion_sigma")
        return self


class DeviationResult(BaseModel):
    """A reading scored against its baseline (docs/05 §5).

    `confidence_calibrated` is False in v1: confidence is a heuristic and must not be
    presented as calibrated until the calibration harness lands (docs/05 §5, §9; T5.2).
    `baseline_ref` is filled by the state engine once the BaselineNode is persisted.
    """

    reading_id: UUID
    patient_id: UUID
    metric_code: MetricCode
    context: MeasurementContext
    z_robust: float
    direction: DeviationDirection
    magnitude: DeviationMagnitude
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_calibrated: bool = False
    is_population_fallback: bool
    baseline_availability: BaselineAvailability
    baseline_ref: UUID | None = None
