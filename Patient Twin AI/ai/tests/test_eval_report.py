"""Aggregated eval report runs every harness and logs gaps (docs/11 §4; T5.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ai.eval_datasets.ppg_dalia import ppg_dalia_available
from ai.eval_datasets.wesad import wesad_available
from ai.eval_report import Blocker, EvalReport, build_report
from scripts.run_eval import _main

_NOW = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
_WESAD_ROOT = Path("datasets/WESAD")
_PPG_ROOT = Path("datasets/PPG-DaLiA")


def test_report_has_every_metric_section() -> None:
    report = build_report(now=_NOW, wesad_root=None, ppg_root=None)
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
    report = build_report(now=_NOW, wesad_root=None, ppg_root=None)
    for section in report.sections:
        assert section.metrics, f"{section.name} produced no metrics"
        for name, value in section.metrics.items():
            assert isinstance(value, float), f"{section.name}.{name} not numeric"


def test_synthetic_smoke_numbers_are_sane() -> None:
    report = build_report(now=_NOW, wesad_root=None, ppg_root=None)
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
    report = build_report(now=_NOW, wesad_root=None, ppg_root=None)
    assert report.gaps, "gaps must be explicitly logged, not silently skipped"
    blockers = {g.blocker for g in report.gaps}
    assert Blocker.GPU_DEP in blockers  # NFR latency/throughput
    assert Blocker.DATASET in blockers  # WESAD/MESA/SHHS wiring
    assert Blocker.CLINICAL_CONFIG in blockers  # UNSET red-flag/lexicon


def test_report_serialises_to_dict() -> None:
    report = build_report(now=_NOW, wesad_root=None, ppg_root=None)
    d = report.to_dict()
    assert set(d) == {"versions", "generated_at", "sections", "gaps"}
    assert d["generated_at"] == _NOW.isoformat()


def test_cli_runs_text_and_json() -> None:
    assert _main([]) == 0
    assert _main(["--json"]) == 0


@pytest.mark.skipif(not ppg_dalia_available(_PPG_ROOT), reason="PPG-DaLiA dataset not on disk")
def test_ppg_dalia_section_wired_when_present() -> None:
    # On a host with the dataset, a real PPG-DaLiA HR section appears (classical DSP on
    # subject-held-out windows) and the PPG DATASET gap is dropped (docs/16 Sprint 10).
    report = build_report(now=_NOW, wesad_root=None, ppg_max_windows_per_subject=50)
    by_name = {s.name: s for s in report.sections}
    assert "ppg_hr" in by_name
    assert by_name["ppg_hr"].dataset == "PPG-DaLiA"
    assert by_name["ppg_hr"].metrics["classical_hr_mae"] > 0
    assert not [g for g in report.gaps if g.metric.startswith("ppg_hr")]


@pytest.mark.skipif(not wesad_available(_WESAD_ROOT), reason="WESAD dataset not on disk")
def test_wesad_wired_replaces_synthetic_and_drops_gap() -> None:
    # On a host with the dataset, the deviation sections carry REAL WESAD numbers and
    # the WESAD-specific DATASET gap is no longer logged (T8.2 DoD).
    report = build_report(now=_NOW, wesad_root=_WESAD_ROOT, wesad_max_subjects=1)
    by_name = {s.name: s for s in report.sections}
    assert by_name["deviation_detection"].dataset == "WESAD"
    assert by_name["deviation_calibration"].dataset == "WESAD"
    assert by_name["deviation_detection"].metrics["n"] > 0
    wesad_gaps = [g for g in report.gaps if "deviation_detection" in g.metric]
    assert not wesad_gaps  # the WESAD gap is closed when real numbers are produced
