"""MAE/RMSE backtest harness — T2.2 DoD: the harness runs and scores forecasts."""

from __future__ import annotations

import pytest

from ai.forecasting.backtest import BacktestResult, backtest, mae, rmse
from ai.forecasting.holt import HoltLinearForecaster

from ._series import daily_series


def test_mae_and_rmse_known_values() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    assert mae([0.0], [2.0]) == 2.0
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx((25 / 2) ** 0.5)
    # RMSE >= MAE always.
    assert rmse([0.0, 0.0], [3.0, 4.0]) >= mae([0.0, 0.0], [3.0, 4.0])


def test_backtest_runs_and_scores() -> None:
    values = [60.0 + 0.2 * i + (0.4 if i % 2 else -0.4) for i in range(40)]
    result = backtest(HoltLinearForecaster(), daily_series(values), horizon_days=3, min_train=5)

    assert isinstance(result, BacktestResult)
    assert result.n_forecasts > 0
    assert result.n_points == result.n_forecasts * 3
    assert result.mae >= 0.0
    assert result.rmse >= result.mae  # holds for any pooled error set


def test_backtest_tracks_clean_linear_trend() -> None:
    # A noise-free linear series should be forecast with small error.
    values = [50.0 + 1.0 * i for i in range(30)]
    result = backtest(HoltLinearForecaster(), daily_series(values), horizon_days=3, min_train=5)
    assert result.mae < 1.0


def test_backtest_rejects_too_short_series() -> None:
    with pytest.raises(ValueError, match="too short"):
        backtest(HoltLinearForecaster(), daily_series([60.0, 61.0]), horizon_days=5, min_train=2)


def test_mae_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        mae([1.0, 2.0], [1.0])
