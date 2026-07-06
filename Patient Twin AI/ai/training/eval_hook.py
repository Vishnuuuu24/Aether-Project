"""Eval hook — score a trained head and prove it wires into the shared harness
(docs/16 Sprint 9; docs/11 §4).

Two jobs: (1) `score_head` computes regression error of a trained head against
held-out targets — the per-run metric a training sprint gates on; (2)
`build_eval_report` re-exports the EXISTING aggregated harness so a trainer scores
through the same path the standing eval uses (no duplicate harness, docs/16 Don't).
Sprint 10 adds a real PPG-DaLiA / WESAD section there once the encoder beats the
classical bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ai.eval_report import EvalReport, build_report
from ai.training.backends import TrainedHead

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


def score_head(head: TrainedHead, features: FloatArray, targets: FloatArray) -> dict[str, float]:
    """MAE / RMSE of a trained head on held-out data (dataset's own GT as truth)."""
    pred = head.predict(features)
    err = pred - targets
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "n": float(targets.size),
    }


def build_eval_report() -> EvalReport:
    """Run the existing aggregated eval harness (the shared scoring path)."""
    return build_report()
