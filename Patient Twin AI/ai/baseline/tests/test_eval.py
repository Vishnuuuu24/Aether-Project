"""Deviation-detection & calibration metrics (docs/11 §1.2; T5.2)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from ai.baseline.eval import (
    LabelledDeviation,
    detection_metrics,
    expected_calibration_error,
    is_flagged,
)
from schemas.baseline import (
    BaselineAvailability,
    DeviationMagnitude,
    DeviationResult,
)
from schemas.psg import DeviationDirection
from schemas.reading import MeasurementContext, MetricCode


def _result(magnitude: DeviationMagnitude, *, confidence: float = 0.9) -> DeviationResult:
    return DeviationResult(
        reading_id=uuid4(),
        patient_id=uuid4(),
        metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING,
        z_robust=0.0,
        direction=DeviationDirection.UP,
        magnitude=magnitude,
        confidence=confidence,
        confidence_calibrated=False,
        is_population_fallback=False,
        baseline_availability=BaselineAvailability.PERSONALISED,
    )


def test_is_flagged_respects_floor() -> None:
    assert not is_flagged(_result(DeviationMagnitude.NORMAL), DeviationMagnitude.MILD)
    assert is_flagged(_result(DeviationMagnitude.MILD), DeviationMagnitude.MILD)
    assert not is_flagged(_result(DeviationMagnitude.MILD), DeviationMagnitude.MODERATE)
    assert is_flagged(_result(DeviationMagnitude.MARKED), DeviationMagnitude.MODERATE)


def test_perfect_detection() -> None:
    labelled = [
        LabelledDeviation(_result(DeviationMagnitude.MARKED), True),
        LabelledDeviation(_result(DeviationMagnitude.MODERATE), True),
        LabelledDeviation(_result(DeviationMagnitude.NORMAL), False),
        LabelledDeviation(_result(DeviationMagnitude.NORMAL), False),
    ]
    m = detection_metrics(labelled)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 0, 0, 2)


def test_mixed_detection_precision_recall() -> None:
    labelled = [
        LabelledDeviation(_result(DeviationMagnitude.MARKED), True),  # TP
        LabelledDeviation(_result(DeviationMagnitude.MILD), False),  # FP
        LabelledDeviation(_result(DeviationMagnitude.NORMAL), True),  # FN
        LabelledDeviation(_result(DeviationMagnitude.NORMAL), False),  # TN
    ]
    m = detection_metrics(labelled)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(0.5)


def test_flag_floor_changes_counts() -> None:
    labelled = [LabelledDeviation(_result(DeviationMagnitude.MILD), True)]
    assert detection_metrics(labelled, flag_at=DeviationMagnitude.MILD).tp == 1
    # Raising the floor unflags a MILD deviation → it becomes a false negative.
    assert detection_metrics(labelled, flag_at=DeviationMagnitude.MODERATE).fn == 1


def test_calibration_perfect_confidence_zero_ece() -> None:
    # Correct predictions all carry confidence 1.0 → mean_conf == accuracy == 1.0.
    labelled = [
        LabelledDeviation(_result(DeviationMagnitude.MARKED, confidence=1.0), True),
        LabelledDeviation(_result(DeviationMagnitude.NORMAL, confidence=1.0), False),
    ]
    cal = expected_calibration_error(labelled)
    assert cal.ece == pytest.approx(0.0)


def test_calibration_overconfident_positive_ece() -> None:
    # Confidence 1.0 but always wrong → ECE == 1.0.
    labelled = [
        LabelledDeviation(_result(DeviationMagnitude.NORMAL, confidence=1.0), True),
        LabelledDeviation(_result(DeviationMagnitude.MARKED, confidence=1.0), False),
    ]
    cal = expected_calibration_error(labelled)
    assert cal.ece == pytest.approx(1.0)


def test_empty_inputs_rejected() -> None:
    with pytest.raises(ValueError, match="no labelled"):
        detection_metrics([])
    with pytest.raises(ValueError, match="no labelled"):
        expected_calibration_error([])
