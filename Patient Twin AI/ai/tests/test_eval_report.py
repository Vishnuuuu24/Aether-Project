"""Aggregated eval report runs every harness and logs gaps (docs/11 §4; T5.2)."""

from __future__ import annotations

from datetime import UTC, datetime

from ai.eval_report import Blocker, EvalReport, build_report
from scripts.run_eval import _main

_NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)


def test_report_has_every_metric_section() -> None:
    report = build_report(now=_NOW)
    assert isinstance(report, EvalReport)
    names = {s.name for s in report.sections}
    assert names == {
        "retrieval",
        "forecast",
        "deviation_detection",
        "deviation_calibration",
        "llm_safety",
    }


def test_every_metric_produces_a_number() -> None:
    report = build_report(now=_NOW)
    for section in report.sections:
        assert section.metrics, f"{section.name} produced no metrics"
        for name, value in section.metrics.items():
            assert isinstance(value, float), f"{section.name}.{name} not numeric"


def test_synthetic_smoke_numbers_are_sane() -> None:
    report = build_report(now=_NOW)
    by_name = {s.name: s for s in report.sections}
    # Clean synthetic separation → the deviation detector should be near-perfect.
    assert by_name["deviation_detection"].metrics["f1"] > 0.9
    # Real Policy gate → mechanical safety guarantees hold.
    safety = by_name["llm_safety"].metrics
    assert safety["grounding_rate"] == 1.0
    assert safety["hallucination_rate"] == 0.0
    assert safety["red_flag_recall"] == 1.0
    assert safety["policy_coverage"] == 1.0


def test_gaps_are_logged_with_blockers() -> None:
    report = build_report(now=_NOW)
    assert report.gaps, "gaps must be explicitly logged, not silently skipped"
    blockers = {g.blocker for g in report.gaps}
    assert Blocker.GPU_DEP in blockers  # NFR latency/throughput
    assert Blocker.DATASET in blockers  # WESAD/MESA/SHHS wiring
    assert Blocker.CLINICAL_CONFIG in blockers  # UNSET red-flag/lexicon


def test_report_serialises_to_dict() -> None:
    report = build_report(now=_NOW)
    d = report.to_dict()
    assert set(d) == {"versions", "generated_at", "sections", "gaps"}
    assert d["generated_at"] == _NOW.isoformat()


def test_cli_runs_text_and_json() -> None:
    assert _main([]) == 0
    assert _main(["--json"]) == 0
