"""Synthetic daily MetricSeries builders for forecasting tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from schemas.forecast import MetricSeries, SeriesPoint
from schemas.reading import MeasurementContext, MetricCode

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def daily_series(
    values: Sequence[float],
    *,
    metric_code: MetricCode = MetricCode.HEART_RATE,
    context: MeasurementContext = MeasurementContext.RESTING,
) -> MetricSeries:
    return MetricSeries(
        patient_id=uuid4(),
        metric_code=metric_code,
        context=context,
        points=[SeriesPoint(ts=_BASE + timedelta(days=i), value=v) for i, v in enumerate(values)],
    )
