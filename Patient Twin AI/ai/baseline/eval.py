"""Deviation-detection & calibration eval harness (docs/11 §1.2; T5.2).

Dataset-agnostic: it scores a set of `DeviationResult`s (from any
`BaselineEngine`) against binary ground-truth labels (is this reading inside a
labelled abnormal/event window?). This is the metric layer — an offline dataset
adapter (WESAD stress, MESA/SHHS sleep events, …) feeds `LabelledDeviation`s in.

Metrics:
  - detection: precision / recall / F1 of flags vs. labels, at a magnitude floor.
  - calibration: reliability bins + Expected Calibration Error over the engine's
    confidence (docs/11 bar: ECE ≤ 0.1). Note v1 confidence is a heuristic
    (`confidence_calibrated=False`); this harness is exactly how we measure whether
    it deserves that label.

Robustness and fallback-honesty (docs/11 §1.2) are invariants of the engine
itself and are asserted in `ai/baseline/tests`, not scored here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from schemas.baseline import MAGNITUDE_RANK, DeviationMagnitude, DeviationResult


@dataclass(frozen=True)
class LabelledDeviation:
    """One scored reading paired with its ground-truth label."""

    result: DeviationResult
    is_abnormal: bool


@dataclass(frozen=True)
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    n: int
    flag_at: DeviationMagnitude


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    n: int
    mean_confidence: float
    empirical_accuracy: float


@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    bins: tuple[CalibrationBin, ...]
    n: int


def is_flagged(result: DeviationResult, flag_at: DeviationMagnitude) -> bool:
    """A reading is 'flagged' when its magnitude is at least the floor (default: any
    non-normal deviation)."""
    return MAGNITUDE_RANK[result.magnitude] >= MAGNITUDE_RANK[flag_at]


def detection_metrics(
    labelled: Sequence[LabelledDeviation],
    *,
    flag_at: DeviationMagnitude = DeviationMagnitude.MILD,
) -> DetectionMetrics:
    if not labelled:
        raise ValueError("no labelled deviations to score")
    tp = fp = fn = tn = 0
    for item in labelled:
        flagged = is_flagged(item.result, flag_at)
        if flagged and item.is_abnormal:
            tp += 1
        elif flagged and not item.is_abnormal:
            fp += 1
        elif not flagged and item.is_abnormal:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n=len(labelled),
        flag_at=flag_at,
    )


def expected_calibration_error(
    labelled: Sequence[LabelledDeviation],
    *,
    flag_at: DeviationMagnitude = DeviationMagnitude.MILD,
    n_bins: int = 10,
) -> CalibrationResult:
    """Reliability of the confidence score: bin by confidence, compare each bin's
    mean confidence to how often the flag decision was actually correct.

    'Correct' = the flag decision matched the label (docs/11: a 0.7 confidence
    should be right ~70% of the time). ECE is the sample-weighted mean absolute
    gap across bins.
    """
    if not labelled:
        raise ValueError("no labelled deviations to score")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for item in labelled:
        conf = item.result.confidence
        correct = is_flagged(item.result, flag_at) == item.is_abnormal
        # conf == 1.0 lands in the last bin, not a phantom (n_bins)th one.
        idx = min(int(conf * n_bins), n_bins - 1)
        buckets[idx].append((conf, correct))

    bins: list[CalibrationBin] = []
    ece = 0.0
    total = len(labelled)
    for b, bucket in enumerate(buckets):
        if not bucket:
            continue
        n = len(bucket)
        mean_conf = sum(c for c, _ in bucket) / n
        acc = sum(1 for _, ok in bucket if ok) / n
        ece += (n / total) * abs(mean_conf - acc)
        bins.append(
            CalibrationBin(
                lower=b / n_bins,
                upper=(b + 1) / n_bins,
                n=n,
                mean_confidence=mean_conf,
                empirical_accuracy=acc,
            )
        )
    return CalibrationResult(ece=ece, bins=tuple(bins), n=total)
