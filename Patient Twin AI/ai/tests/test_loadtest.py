"""NFR load/latency harness runs and reports (docs/01 NFR-1/NFR-2; T5.3).

Asserts the harness *works* and produces numbers; the actual NFR pass/fail on the
Mac is not asserted (throughput is machine-dependent; NFR-1's full path is
H200-only). NFR-2 is CPU-bound and comfortably exceeds target on any modern host.
"""

from __future__ import annotations

import pytest

from scripts.loadtest import (
    NFR2_TARGET_READINGS_PER_S,
    measure_copilot_overhead,
    measure_ingestion_throughput,
)
from scripts.loadtest import _main as loadtest_main


def test_ingestion_throughput_processes_all_and_reports_rate() -> None:
    r = measure_ingestion_throughput(500)
    assert r.n_processed == 500
    assert r.readings_per_s > 0
    assert r.target == NFR2_TARGET_READINGS_PER_S
    # CPU-bound normalisation clears the 50 readings/s bar with huge margin.
    assert r.meets_target


def test_copilot_overhead_reports_percentiles() -> None:
    r = measure_copilot_overhead(20)
    assert r.n == 20
    assert r.p50_ms >= 0.0
    assert r.p95_ms >= r.p50_ms
    assert "GPU-DEP" in r.note or "‹GPU-DEP›" in r.note


def test_harness_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError, match="n must be"):
        measure_ingestion_throughput(0)
    with pytest.raises(ValueError, match="n must be"):
        measure_copilot_overhead(0)


def test_cli_runs() -> None:
    assert loadtest_main(["--readings", "200", "--iters", "10"]) == 0
