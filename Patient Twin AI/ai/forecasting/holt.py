"""HoltLinearForecaster — v1 `Forecaster` (docs/05 §7).

Double exponential smoothing (Holt's linear trend): maintain a level and a trend,
forecast `level + h*trend` h steps ahead. Prediction intervals come from the
in-sample one-step residual spread, widened with the square root of the horizon
(uncertainty accumulates). Intervals are a heuristic spread — NOT calibrated
(calibration is a later concern, docs/05 §5/§9).

Decision support only: this extrapolates a metric trajectory, never a disease.
"""

from __future__ import annotations

import statistics
from math import sqrt

from schemas.forecast import Forecast, MetricSeries

FORECASTER_VERSION = "holt-linear-v1"
_METHOD = "holt_linear"
_Z_95 = 1.96  # normal quantile for a nominal 95% interval (uncalibrated)


class HoltLinearForecaster:
    """Implements the `Forecaster` protocol (docs/02 §6)."""

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        beta: float = 0.1,
        z: float = _Z_95,
        version: str = FORECASTER_VERSION,
    ) -> None:
        if not (0.0 < alpha <= 1.0 and 0.0 <= beta <= 1.0):
            raise ValueError("alpha must be in (0,1] and beta in [0,1]")
        self._alpha = alpha
        self._beta = beta
        self._z = z
        self._version = version

    def forecast(self, series: MetricSeries, *, horizon_days: int) -> Forecast:
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")
        values = [p.value for p in series.points]
        if len(values) < 2:
            raise ValueError("Holt's linear trend needs at least 2 observations")

        level, trend, residuals = _fit(values, self._alpha, self._beta)
        points = [level + step * trend for step in range(1, horizon_days + 1)]

        sigma = statistics.stdev(residuals) if len(residuals) >= 2 else 0.0
        intervals = [
            (point - self._z * sigma * sqrt(step), point + self._z * sigma * sqrt(step))
            for step, point in enumerate(points, start=1)
        ]

        return Forecast(
            patient_id=series.patient_id,
            metric_code=series.metric_code,
            context=series.context,
            horizon_days=horizon_days,
            points=points,
            intervals=intervals,
            method=_METHOD,
            forecaster_version=self._version,
        )


def _fit(values: list[float], alpha: float, beta: float) -> tuple[float, float, list[float]]:
    """Run Holt's linear smoothing; return final (level, trend) and the in-sample
    one-step-ahead residuals used to size prediction intervals.
    """
    level = values[0]
    trend = values[1] - values[0]
    residuals: list[float] = []
    for actual in values[1:]:
        forecast_one = level + trend  # one-step-ahead before seeing `actual`
        residuals.append(actual - forecast_one)
        prev_level = level
        level = alpha * actual + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
    return level, trend, residuals
