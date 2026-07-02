"""Rolling-origin backtest harness for forecasts — MAE / RMSE (docs/05 §9; T2.2 DoD).

For each origin `k` from `min_train` onward, fit on `series[:k]`, forecast `horizon_days`
steps, and compare each step to the held-out actuals `series[k : k+horizon_days]`.
Errors are pooled across all origins and steps into MAE and RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from ai.interfaces.forecaster import Forecaster
from schemas.forecast import MetricSeries


def mae(actual: list[float], predicted: list[float]) -> float:
    _check(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    _check(actual, predicted)
    return sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / len(actual))


@dataclass(frozen=True)
class BacktestResult:
    mae: float
    rmse: float
    n_forecasts: int  # number of origins evaluated
    n_points: int  # number of (predicted, actual) pairs pooled


def backtest(
    forecaster: Forecaster,
    series: MetricSeries,
    *,
    horizon_days: int,
    min_train: int = 2,
) -> BacktestResult:
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if min_train < 2:
        raise ValueError("min_train must be >= 2 (Holt needs two observations)")

    points = series.points
    actuals: list[float] = []
    preds: list[float] = []
    origins = 0
    last_origin = len(points) - horizon_days  # inclusive upper bound for k
    for k in range(min_train, last_origin + 1):
        train = MetricSeries(
            patient_id=series.patient_id,
            metric_code=series.metric_code,
            context=series.context,
            points=points[:k],
        )
        forecast = forecaster.forecast(train, horizon_days=horizon_days)
        for step in range(horizon_days):
            preds.append(forecast.points[step])
            actuals.append(points[k + step].value)
        origins += 1

    if not actuals:
        raise ValueError("series too short for the requested horizon and min_train")
    return BacktestResult(
        mae=mae(actuals, preds),
        rmse=rmse(actuals, preds),
        n_forecasts=origins,
        n_points=len(actuals),
    )


def _check(actual: list[float], predicted: list[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must be the same length")
    if not actual:
        raise ValueError("cannot score an empty series")
