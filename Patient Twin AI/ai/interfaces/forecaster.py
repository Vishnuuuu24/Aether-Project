"""Stable `Forecaster` interface (docs/02 §6, docs/05 §7).

v1 impl: `ai.forecasting.HoltLinearForecaster` (exponential-smoothing / Holt-style).
DEFERRED impl: a temporal foundation model — a new implementation of THIS protocol,
never a new call site. Forecasts predict metric trajectories, never disease.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas.forecast import Forecast, MetricSeries


@runtime_checkable
class Forecaster(Protocol):
    def forecast(self, series: MetricSeries, *, horizon_days: int) -> Forecast: ...
