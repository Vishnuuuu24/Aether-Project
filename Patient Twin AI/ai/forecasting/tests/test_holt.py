"""HoltLinearForecaster — T2.2 DoD: forecasts (point + interval) for resting HR & sleep."""

from __future__ import annotations

import pytest

from ai.forecasting.holt import FORECASTER_VERSION, HoltLinearForecaster
from schemas.reading import MeasurementContext, MetricCode

from ._series import daily_series


def test_resting_hr_forecast_has_points_and_intervals() -> None:
    # 30 days trending gently up with alternating noise so residual sigma > 0.
    values = [60.0 + 0.1 * i + (0.5 if i % 2 else -0.5) for i in range(30)]
    forecast = HoltLinearForecaster().forecast(daily_series(values), horizon_days=7)

    assert forecast.horizon_days == 7
    assert len(forecast.points) == 7
    assert len(forecast.intervals) == 7
    assert forecast.method == "holt_linear"
    assert forecast.forecaster_version == FORECASTER_VERSION
    # Upward trend is extrapolated; every interval brackets its point.
    assert forecast.points[-1] > forecast.points[0]
    for point, (low, high) in zip(forecast.points, forecast.intervals, strict=True):
        assert low <= point <= high


def test_intervals_widen_with_horizon() -> None:
    values = [7.0 + (0.3 if i % 2 else -0.3) for i in range(20)]  # noisy sleep hours
    forecast = HoltLinearForecaster().forecast(
        daily_series(values, metric_code=MetricCode.SLEEP, context=MeasurementContext.ASLEEP),
        horizon_days=7,
    )
    widths = [high - low for low, high in forecast.intervals]
    assert all(b >= a for a, b in zip(widths, widths[1:], strict=False))  # non-decreasing
    assert widths[-1] > widths[0]  # uncertainty accumulates


def test_constant_series_has_flat_forecast_and_zero_width() -> None:
    forecast = HoltLinearForecaster().forecast(daily_series([60.0] * 10), horizon_days=5)
    assert all(p == pytest.approx(60.0) for p in forecast.points)
    assert all(high == pytest.approx(low) for low, high in forecast.intervals)  # sigma == 0


def test_needs_two_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        HoltLinearForecaster().forecast(daily_series([60.0]), horizon_days=7)


def test_rejects_nonpositive_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        HoltLinearForecaster().forecast(daily_series([60.0, 61.0]), horizon_days=0)


def test_rejects_bad_smoothing_params() -> None:
    with pytest.raises(ValueError, match="alpha"):
        HoltLinearForecaster(alpha=0.0)
