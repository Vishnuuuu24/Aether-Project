"""Forecast contracts (docs/02 §6, docs/05 §7).

`MetricSeries` is the input to `Forecaster.forecast`; `Forecast` is its output —
point trajectory + prediction interval per step. Forecasts predict *metric
trajectories*, never disease (docs/05 §7). The Patient State Engine maps a `Forecast`
to a versioned `ForecastNode` (schemas.psg) when it commits it.

CLAUDE.md: contracts live in schemas/ only.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from .reading import MeasurementContext, MetricCode


class SeriesPoint(BaseModel):
    ts: datetime
    value: float

    @model_validator(mode="after")
    def _tz_aware(self) -> SeriesPoint:
        if self.ts.tzinfo is None:
            raise ValueError("series point ts must be timezone-aware")
        return self


class MetricSeries(BaseModel):
    """A time-ordered value series for one `(patient, metric, context)` — typically a
    daily-aggregated series (or the baseline trend) the forecaster extrapolates.
    """

    patient_id: UUID
    metric_code: MetricCode
    context: MeasurementContext
    points: list[SeriesPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _monotonic(self) -> MetricSeries:
        for earlier, later in zip(self.points, self.points[1:], strict=False):
            if later.ts < earlier.ts:
                raise ValueError("series points must be in non-decreasing ts order")
        return self


class Forecast(BaseModel):
    """A short-horizon forecast: one point + one (lower, upper) interval per step
    (docs/05 §7). `horizon_days` steps at daily granularity by default.
    """

    patient_id: UUID
    metric_code: MetricCode
    context: MeasurementContext
    horizon_days: int = Field(ge=1)
    points: list[float]
    intervals: list[tuple[float, float]]
    method: str
    forecaster_version: str
    generated_at: datetime | None = None

    @model_validator(mode="after")
    def _shape(self) -> Forecast:
        if len(self.points) != self.horizon_days:
            raise ValueError("points length must equal horizon_days")
        if len(self.intervals) != self.horizon_days:
            raise ValueError("intervals length must equal horizon_days")
        for low, high in self.intervals:
            if high < low:
                raise ValueError("interval upper bound must be >= lower bound")
        return self
