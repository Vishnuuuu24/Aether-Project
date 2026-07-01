"""PSG projection is the ONLY thing the LLM sees, and it must carry no raw
signals and no reading-level data (docs/04 §5, CLAUDE.md principle 2).
"""

from __future__ import annotations

from datetime import UTC, datetime

from schemas import MeasurementContext, MetricCode, PSGProjection
from schemas.psg import BaselineSummary, DeviationDirection, DeviationSummary, VersionStamp


def make_projection() -> PSGProjection:
    return PSGProjection(
        patient_age_years=40,
        patient_sex_at_birth="male",
        baselines=[
            BaselineSummary(
                metric_code=MetricCode.HEART_RATE,
                context=MeasurementContext.RESTING,
                center=58.0,
                dispersion=4.0,
                confidence=0.9,
                is_population_fallback=False,
            )
        ],
        recent_deviations=[
            DeviationSummary(
                metric_code=MetricCode.HEART_RATE,
                direction=DeviationDirection.UP,
                magnitude=2.5,
                z_robust=3.1,
                confidence=0.8,
                ts=datetime(2026, 6, 1, tzinfo=UTC),
            )
        ],
        as_of=datetime(2026, 6, 1, tzinfo=UTC),
        consent_scope=["vitals"],
        versions=VersionStamp(baseline_engine="b1", ruleset="r1", prompt="p1", model="m1"),
    )


def test_projection_model_has_no_reading_level_fields() -> None:
    fields = set(PSGProjection.model_fields)
    # No raw window pointers, no reading arrays, no per-sample values.
    assert "raw_ref" not in fields
    assert "readings" not in fields
    assert "value" not in fields


def test_projection_json_contains_no_raw_ref() -> None:
    payload = make_projection().model_dump_json()
    assert "raw_ref" not in payload


def test_baseline_summary_is_center_dispersion_only() -> None:
    # A summary exposes the baseline's center/dispersion — never the raw sample
    # window it was computed from.
    dumped = make_projection().model_dump()
    baseline_keys = set(dumped["baselines"][0])
    assert baseline_keys == {
        "metric_code",
        "context",
        "center",
        "dispersion",
        "confidence",
        "is_population_fallback",
    }


def test_projection_roundtrip() -> None:
    proj = make_projection()
    assert PSGProjection.model_validate_json(proj.model_dump_json()) == proj
