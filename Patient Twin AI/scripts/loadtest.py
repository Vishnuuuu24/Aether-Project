"""NFR load/latency harness (docs/01 NFR-1/NFR-2; docs/10 T5.3).

Two NFRs, two very different measurability stories on the Mac:

  NFR-2  Ingestion ≥ 50 readings/s/instance without backpressure loss.
         CPU-bound normalisation — MEASURABLE HERE. This harness produces a real
         number and a pass/fail against the target.

  NFR-1  Copilot p95 end-to-end latency ≤ 6 s for a grounded answer.  ‹GPU-DEP›
         Dominated by LLM inference on the H200; per CLAUDE.md we do NOT load-test
         the model on the Mac (single-tenant, ~20–50 tok/s — not representative).
         This harness measures only the DETERMINISTIC overhead (retrieve + policy,
         LLM stubbed) as a floor; the real p95 must be measured on the H200 slice.

    python -m scripts.loadtest                 # both, human-readable
    python -m scripts.loadtest --readings 5000 # NFR-2 with a bigger batch
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter
from uuid import uuid4

from core.versioning import VersionRegistry
from schemas.consent import ConsentScope
from schemas.output_contract import ProposedOutput
from schemas.psg import PSGProjection
from schemas.reading import MeasurementContext, MetricCode
from schemas.retrieval import EvidenceChunk, RetrievalScope
from services.copilot_service.eval import default_safety_cases
from services.copilot_service.orchestrator import Copilot
from services.ingestion_service.normaliser import normalise_batch
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import PolicyRuleSet

NFR2_TARGET_READINGS_PER_S = 50.0
NFR1_TARGET_P95_SECONDS = 6.0  # ‹GPU-DEP› — full path only meaningful on the H200


@dataclass(frozen=True)
class ThroughputResult:
    n_processed: int
    seconds: float
    readings_per_s: float
    target: float
    meets_target: bool


@dataclass(frozen=True)
class LatencyResult:
    n: int
    p50_ms: float
    p95_ms: float
    note: str


def _synthetic_readings(n: int) -> list[dict[str, object]]:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    pid = uuid4()
    return [
        {
            "patient_id": pid,
            "metric_code": MetricCode.HEART_RATE.value,
            "value": 60.0 + (i % 5),
            "unit": "bpm",
            "timestamp": base + timedelta(seconds=i),
            "source_device": "apple_watch_s9",
            "sqi": 0.9,
            "context": MeasurementContext.RESTING.value,
            "ingest_adapter": "csv",
        }
        for i in range(n)
    ]


def measure_ingestion_throughput(n: int = 2000) -> ThroughputResult:
    """NFR-2: real throughput of the ingestion normaliser over `n` readings."""
    if n < 1:
        raise ValueError("n must be >= 1")
    batch = _synthetic_readings(n)
    start = perf_counter()
    result = normalise_batch(batch, default_adapter="csv")
    elapsed = perf_counter() - start
    processed = len(result.accepted)
    rate = processed / elapsed if elapsed > 0 else float("inf")
    return ThroughputResult(
        n_processed=processed,
        seconds=elapsed,
        readings_per_s=rate,
        target=NFR2_TARGET_READINGS_PER_S,
        meets_target=rate >= NFR2_TARGET_READINGS_PER_S,
    )


class _StubGateway:
    def __init__(self, proposal: ProposedOutput) -> None:
        self._proposal = proposal

    def propose(
        self,
        *,
        query: str,
        projection: PSGProjection,
        evidence: list[EvidenceChunk],
        locale: str = "en",
    ) -> ProposedOutput:
        return self._proposal


class _StubRetriever:
    def __init__(self, evidence: list[EvidenceChunk]) -> None:
        self._evidence = evidence

    def search(self, query: str, scope: RetrievalScope, *, k: int = 10) -> list[EvidenceChunk]:
        return self._evidence


def measure_copilot_overhead(n: int = 200) -> LatencyResult:
    """NFR-1 floor: deterministic retrieve+policy overhead with the LLM STUBBED.

    This is NOT the NFR-1 number — it excludes model inference (‹GPU-DEP›). It
    bounds how much of the 6 s budget the deterministic path consumes.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    case = default_safety_cases()[0]  # the grounded 'approved' case
    proposal_proj: PSGProjection = case.projection
    copilot = Copilot(
        retriever=_StubRetriever(case.evidence),
        gateway=_StubGateway(case.proposal),
        policy=PolicyEngine(PolicyRuleSet()),
        versions=VersionRegistry.from_env().current(),
    )
    now = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
    samples_ms: list[float] = []
    for _ in range(n):
        start = perf_counter()
        copilot.answer(
            patient_id=case.patient_id,
            projection=proposal_proj,
            query="why is my resting heart rate up?",
            consented_scopes=[ConsentScope.COPILOT, ConsentScope.VITALS],
            now=now,
        )
        samples_ms.append((perf_counter() - start) * 1000.0)
    samples_ms.sort()
    return LatencyResult(
        n=n,
        p50_ms=_percentile(samples_ms, 50),
        p95_ms=_percentile(samples_ms, 95),
        note="deterministic overhead only; LLM inference budget is ‹GPU-DEP› (H200)",
    )


def _percentile(sorted_samples: list[float], pct: float) -> float:
    if not sorted_samples:
        return 0.0
    k = max(0, min(len(sorted_samples) - 1, int(round((pct / 100.0) * (len(sorted_samples) - 1)))))
    return sorted_samples[k]


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the NFR load/latency harness.")
    parser.add_argument("--readings", type=int, default=2000, help="NFR-2 batch size")
    parser.add_argument("--iters", type=int, default=200, help="NFR-1 overhead iterations")
    args = parser.parse_args(argv)

    tput = measure_ingestion_throughput(args.readings)
    print("NFR-2  ingestion throughput (measured on this host):")
    print(f"    processed {tput.n_processed} readings in {tput.seconds * 1000:.1f} ms")
    print(f"    {tput.readings_per_s:,.0f} readings/s  (target ≥ {tput.target:.0f})")
    print(f"    -> {'PASS' if tput.meets_target else 'BELOW TARGET'}\n")

    lat = measure_copilot_overhead(args.iters)
    print("NFR-1  copilot latency  [‹GPU-DEP› — full path is H200-only]:")
    print(f"    deterministic overhead p50={lat.p50_ms:.2f} ms  p95={lat.p95_ms:.2f} ms")
    print(f"    target (full path) p95 ≤ {NFR1_TARGET_P95_SECONDS:.0f} s")
    print(f"    note: {lat.note}")
    print("    -> DEFERRED: measure end-to-end p95 with the real LLM on the H200 slice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
