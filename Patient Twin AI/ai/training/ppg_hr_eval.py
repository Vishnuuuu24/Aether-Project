"""Head-to-head PPG→HR evaluation on held-out subjects (docs/16 Sprint 10 DoD).

The DoD's literal bar is "the DL path matches or beats the **classical** HR pipeline".
This scores both on the SAME subject-held-out PPG-DaLiA windows: the classical DSP
`WaveformFeatureExtractor` (systolic-peak detection) and, when a trained checkpoint is
supplied, the learned conv encoder. NumPy-only — no MLX needed to evaluate.

Reused by the training CLI (to print the comparison) and by `ai.eval_report` (to add
the `dataset="PPG-DaLiA"` section). Coverage is reported for the classical path since
peak detection can fail on a motion-corrupted window; the encoder always emits a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ai.eval_datasets.ppg_dalia import load_ppg_dalia_hr_signal_windows
from ai.features.waveform_extractor import WaveformFeatureExtractor
from ai.training.encoder_model import EncoderWeights, predict_hr
from ai.training.splits import subject_held_out_split
from schemas.features import RawWaveform, SignalWindow, WaveformKind
from schemas.reading import MeasurementContext, MetricCode

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

_TS = datetime(2026, 1, 1, tzinfo=UTC)
_DEFAULT_SEED = 20260101
_DEFAULT_VAL_FRACTION = 0.25


def classical_hr_predictions(
    signals: FloatArray,
    sample_rate_hz: float,
    *,
    extractor: WaveformFeatureExtractor | None = None,
) -> FloatArray:
    """Per-window classical HR (bpm); NaN where peak detection produced no HR."""
    ext = extractor or WaveformFeatureExtractor()
    pid = uuid4()
    preds = np.full(len(signals), np.nan, dtype=np.float64)
    for i, seg in enumerate(signals):
        window = SignalWindow(
            patient_id=pid,
            metric_code=MetricCode.HEART_RATE,
            context=MeasurementContext.RESTING,
            window_start=_TS,
            window_end=_TS,
            waveform=RawWaveform(
                kind=WaveformKind.PPG, sample_rate_hz=sample_rate_hz, samples=seg.tolist()
            ),
        )
        hr = ext.extract(window).features.get("heart_rate_bpm")
        if hr is not None:
            preds[i] = hr
    return preds


def _score(pred: FloatArray, target: FloatArray) -> dict[str, float]:
    mask = np.isfinite(pred)
    if not mask.any():
        return {"mae": float("nan"), "rmse": float("nan"), "coverage": 0.0, "n": 0.0}
    err = pred[mask] - target[mask]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "coverage": float(mask.mean()),
        "n": float(mask.sum()),
    }


@dataclass(frozen=True)
class HeadToHead:
    held_out_subjects: tuple[str, ...]
    n_val: int
    classical: dict[str, float]
    encoder: dict[str, float] | None  # None when no checkpoint supplied

    @property
    def encoder_beats_classical(self) -> bool | None:
        if self.encoder is None:
            return None
        return self.encoder["mae"] < self.classical["mae"]


def evaluate_holdout(
    root: Path,
    *,
    weights: EncoderWeights | None = None,
    seed: int = _DEFAULT_SEED,
    val_fraction: float = _DEFAULT_VAL_FRACTION,
    max_windows_per_subject: int | None = None,
) -> HeadToHead:
    """Score classical (always) and encoder (if `weights` given) on held-out subjects.

    Uses the same subject-held-out split parameters as training so the comparison is
    apples-to-apples on subjects the encoder never saw.
    """
    windows = load_ppg_dalia_hr_signal_windows(
        root, max_windows_per_subject=max_windows_per_subject
    )
    _, val_idx, _, val_subj = subject_held_out_split(
        windows.subject_ids, val_fraction=val_fraction, seed=seed
    )
    sig_va, y_va = windows.signals[val_idx], windows.targets[val_idx]
    classical = _score(classical_hr_predictions(sig_va, windows.sample_rate_hz), y_va)
    encoder = _score(predict_hr(weights, sig_va), y_va) if weights is not None else None
    return HeadToHead(
        held_out_subjects=val_subj, n_val=int(len(y_va)), classical=classical, encoder=encoder
    )
