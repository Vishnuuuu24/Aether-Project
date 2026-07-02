"""v1 forecasting (docs/05 §7).

`HoltLinearForecaster` implements the stable `Forecaster` interface with double
exponential smoothing (level + trend) and residual-based prediction intervals.
`backtest` is the rolling-origin MAE/RMSE harness.
"""

from .backtest import BacktestResult, backtest, mae, rmse
from .holt import FORECASTER_VERSION, HoltLinearForecaster

__all__ = [
    "FORECASTER_VERSION",
    "BacktestResult",
    "HoltLinearForecaster",
    "backtest",
    "mae",
    "rmse",
]
