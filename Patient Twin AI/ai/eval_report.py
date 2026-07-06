"""Aggregated evaluation report (docs/11 §4; T5.2).

Runs every §1 harness through the REAL engines and produces one versioned report.
Where an offline clinical dataset isn't wired into the harness yet, the metric
still produces a number on a small built-in SMOKE sample (clearly labelled
`dataset="synthetic"`) and the real-dataset wiring is recorded as an explicit
`EvalGap` — never silently skipped. GPU-dependent NFRs and UNSET clinical config
are logged the same way. This is the "all metrics produce numbers on offline
datasets; safety thresholds met or gaps logged" DoD, honestly.

The `synthetic` samples are smoke tests of the *harness*, not clinical results.
Real numbers require the offline-dataset adapters listed in `gaps`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from uuid import uuid4

from ai.baseline.eval import (
    LabelledDeviation,
    detection_metrics,
    expected_calibration_error,
)
from ai.baseline.statistical import StatisticalBaselineEngine
from ai.eval_datasets.ppg_dalia import ppg_dalia_available
from ai.eval_datasets.wesad import load_wesad_labelled_deviations, wesad_available
from ai.features.sqi import SqiGate
from ai.forecasting.backtest import backtest
from ai.forecasting.holt import HoltLinearForecaster
from ai.retrieval.embedder import HashEmbedder
from ai.retrieval.eval import EvalQuery, evaluate
from ai.retrieval.hybrid import HybridRetriever
from ai.retrieval.reranker import LexicalReranker
from ai.retrieval.vector_store import InMemoryVectorStore
from core.versioning import VersionRegistry
from core.versioning.registry import VersionSet
from schemas.forecast import MetricSeries, SeriesPoint
from schemas.reading import MeasurementContext, MetricCode, Reading
from schemas.retrieval import RetrievalScope
from schemas.vector import VectorPayload, VectorSourceType
from services.copilot_service.eval import default_safety_cases, evaluate_safety
from services.policy_engine.engine import PolicyEngine
from services.policy_engine.rules import PolicyRuleSet

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


class Blocker(str, Enum):
    GPU_DEP = "gpu_dependency"  # needs the H200 slice; not measurable on the Mac
    DATASET = "dataset_not_wired"  # offline dataset present but no harness adapter yet
    CLINICAL_CONFIG = "clinical_config_unset"  # gated on clinician-provided config


@dataclass(frozen=True)
class MetricSection:
    name: str
    dataset: str  # "synthetic" (harness smoke) | dataset name once wired
    metrics: dict[str, float]


@dataclass(frozen=True)
class EvalGap:
    metric: str
    blocker: Blocker
    detail: str


@dataclass
class EvalReport:
    versions: dict[str, str]
    generated_at: str
    sections: list[MetricSection] = field(default_factory=list)
    gaps: list[EvalGap] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "versions": self.versions,
            "generated_at": self.generated_at,
            "sections": [asdict(s) for s in self.sections],
            "gaps": [asdict(g) for g in self.gaps],
        }


# -- synthetic smoke samples (exercise the harness; NOT clinical results) --------


def _synthetic_retrieval_section() -> MetricSection:
    corpus = [
        VectorPayload(
            source_type=VectorSourceType.KB_PASSAGE,
            chunk_text=text,
            chunk_index=i,
            embedding_model="hash-dev",
            timestamp=_BASE,
        )
        for i, text in enumerate(
            [
                "metformin is first-line therapy for type 2 diabetes",
                "atrial fibrillation requires anticoagulation",
                "hypertension managed with lifestyle and medication",
            ]
        )
    ]
    retriever = HybridRetriever(
        corpus,
        embedder=HashEmbedder(),
        reranker=LexicalReranker(),
        vector_store=InMemoryVectorStore(),
    )
    queries = [
        EvalQuery(
            query="type 2 diabetes treatment",
            scope=RetrievalScope(include_kb=True),
            relevant_ids=frozenset({corpus[0].chunk_id}),
        ),
        EvalQuery(
            query="anticoagulation for atrial fibrillation",
            scope=RetrievalScope(include_kb=True),
            relevant_ids=frozenset({corpus[1].chunk_id}),
        ),
    ]
    r = evaluate(retriever, queries, k=3)
    return MetricSection(
        name="retrieval",
        dataset="synthetic",
        metrics={
            "recall_at_k": r.recall_at_k,
            "mrr": r.mrr,
            "ndcg_at_k": r.ndcg_at_k,
            "k": float(r.k),
        },
    )


def _synthetic_forecast_section() -> MetricSection:
    values = [60.0 + 0.2 * i + (0.4 if i % 2 else -0.4) for i in range(40)]
    series = MetricSeries(
        patient_id=uuid4(),
        metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING,
        points=[SeriesPoint(ts=_BASE + timedelta(days=i), value=v) for i, v in enumerate(values)],
    )
    r = backtest(HoltLinearForecaster(), series, horizon_days=3, min_train=5)
    return MetricSection(
        name="forecast",
        dataset="synthetic",
        metrics={"mae": r.mae, "rmse": r.rmse, "n_forecasts": float(r.n_forecasts)},
    )


def _synthetic_deviation_labelled() -> list[LabelledDeviation]:
    pid = uuid4()
    engine = StatisticalBaselineEngine(gate=SqiGate({"heart_rate": 0.5}), patient_id=pid)
    cycle = [58.0, 59.0, 60.0, 61.0, 62.0]

    def reading(value: float, *, hours: float) -> Reading:
        return Reading(
            patient_id=pid,
            metric_code=MetricCode.HEART_RATE,
            value=value,
            unit="bpm",
            timestamp=_BASE + timedelta(hours=hours),
            source_device="apple_watch_s9",
            sqi=0.9,
            context=MeasurementContext.RESTING,
            ingest_adapter="csv",
        )

    for i in range(60):  # personalise the baseline over ~20 days
        engine.update(reading(cycle[i % len(cycle)], hours=i * 8))

    labelled: list[LabelledDeviation] = []
    for i in range(20):  # normal readings — label False
        labelled.append(
            LabelledDeviation(engine.score(reading(cycle[i % 5], hours=500 + i)), False)
        )
    for i in range(20):  # clearly abnormal readings — label True
        labelled.append(LabelledDeviation(engine.score(reading(85.0, hours=600 + i)), True))
    return labelled


def _deviation_sections(
    dataset: str, labelled: list[LabelledDeviation]
) -> tuple[MetricSection, MetricSection]:
    """Detection + calibration sections over any labelled-deviation set (synthetic or
    a real offline dataset — the metric layer is dataset-agnostic, docs/11 §1.2)."""
    det = detection_metrics(labelled)
    cal = expected_calibration_error(labelled)
    detection = MetricSection(
        name="deviation_detection",
        dataset=dataset,
        metrics={"precision": det.precision, "recall": det.recall, "f1": det.f1, "n": float(det.n)},
    )
    calibration = MetricSection(
        name="deviation_calibration",
        dataset=dataset,
        metrics={"ece": cal.ece, "n": float(cal.n)},
    )
    return detection, calibration


def _synthetic_deviation_sections() -> tuple[MetricSection, MetricSection]:
    return _deviation_sections("synthetic", _synthetic_deviation_labelled())


def _safety_section() -> MetricSection:
    engine = PolicyEngine(PolicyRuleSet())  # UNSET clinical config → inert content checks
    m = evaluate_safety(engine, default_safety_cases())
    return MetricSection(
        name="llm_safety",
        dataset="authored",  # authored structural cases, run through the real Policy gate
        metrics={
            "grounding_rate": m.grounding_rate,
            "hallucination_rate": m.hallucination_rate,
            "abstention_correctness": m.abstention_correctness,
            "scope_violation_rate": m.scope_violation_rate,
            "red_flag_recall": m.red_flag_recall,
            "policy_coverage": m.policy_coverage,
        },
    )


def _ppg_dalia_hr_section(
    root: Path,
    *,
    encoder_checkpoint: Path | None,
    max_windows_per_subject: int,
) -> MetricSection:
    """Real PPG-DaLiA HR numbers: the classical DSP pipeline always; the learned
    encoder too when a checkpoint is configured (docs/16 Sprint 10)."""
    from ai.training.ppg_hr_eval import evaluate_holdout

    weights = None
    if encoder_checkpoint is not None:
        from ai.training.checkpoints import load_encoder_weights

        weights = load_encoder_weights(encoder_checkpoint)
    h = evaluate_holdout(root, weights=weights, max_windows_per_subject=max_windows_per_subject)
    metrics = {
        "classical_hr_mae": h.classical["mae"],
        "classical_coverage": h.classical["coverage"],
        "n": float(h.n_val),
    }
    if h.encoder is not None:
        metrics["encoder_hr_mae"] = h.encoder["mae"]
    return MetricSection(name="ppg_hr", dataset="PPG-DaLiA", metrics=metrics)


def _gaps(*, wesad_wired: bool, ppg_wired: bool) -> list[EvalGap]:
    gaps: list[EvalGap] = []
    if not ppg_wired:
        gaps.append(
            EvalGap(
                metric="ppg_hr (PPG-DaLiA)",
                blocker=Blocker.DATASET,
                detail=(
                    "Ran without a PPG-DaLiA HR section: the dataset is not present. On a "
                    "host with datasets/PPG-DaLiA/ this section reports classical-DSP HR MAE "
                    "(and the learned encoder's, when a checkpoint is configured) on "
                    "subject-held-out windows (docs/16 Sprint 10; see docs/17 for the run log)."
                ),
            )
        )
    if not wesad_wired:
        gaps.append(
            EvalGap(
                metric="deviation_detection / calibration",
                blocker=Blocker.DATASET,
                detail=(
                    "Ran on synthetic input: the WESAD dataset is not present in this "
                    "environment. On a host with datasets/WESAD/ the classical "
                    "WaveformFeatureExtractor (T8.1) derives HR from raw ECG and this "
                    "section reports real WESAD stress-vs-baseline numbers (T8.2). Further "
                    "datasets (MESA/SHHS sleep, PPG-DaLiA activity, WildPPG noise) remain "
                    "to be adapted."
                ),
            )
        )
    gaps.extend(
        [
            EvalGap(
                metric="forecast interval calibration",
                blocker=Blocker.DATASET,
                detail=(
                    "MAE/RMSE vs naive is covered; empirical prediction-interval coverage "
                    "needs interval-producing forecasts wired over a real longitudinal series."
                ),
            ),
            EvalGap(
                metric="llm_safety red_flag_recall (content patterns)",
                blocker=Blocker.CLINICAL_CONFIG,
                detail=(
                    "Only the always-on structural HIGH-severity-event rule is exercised. "
                    "Configured acute red-flag patterns are UNSET clinical config "
                    "(config/clinical/policy_rules.yaml) — content-specific recall is gated "
                    "on clinician-authored patterns."
                ),
            ),
            EvalGap(
                metric="llm_safety scope_violation_rate (lexicon)",
                blocker=Blocker.CLINICAL_CONFIG,
                detail=(
                    "Enum-closed action vocabulary makes structural scope violations "
                    "impossible, but the prohibited clinical lexicon is UNSET config; the "
                    "gate is proven by test, not yet measured against a real term list."
                ),
            ),
            EvalGap(
                metric="tooling latency (NFR-1) / throughput (NFR-2)",
                blocker=Blocker.GPU_DEP,
                detail=(
                    "End-to-end latency and ingestion throughput are measured on the H200 "
                    "slice only (CLAUDE.md: do not load-test on the Mac). Deferred to T5.3 "
                    "on-server."
                ),
            ),
        ]
    )
    return gaps


# Auto-detected when a caller doesn't pass an explicit root (present on the Mac, absent in CI).
DEFAULT_WESAD_ROOT = Path("datasets/WESAD")
DEFAULT_PPG_DALIA_ROOT = Path("datasets/PPG-DaLiA")


def build_report(
    *,
    versions: VersionSet | None = None,
    now: datetime | None = None,
    wesad_root: Path | None = DEFAULT_WESAD_ROOT,
    wesad_max_subjects: int = 3,
    ppg_root: Path | None = DEFAULT_PPG_DALIA_ROOT,
    ppg_encoder_checkpoint: Path | None = None,
    ppg_max_windows_per_subject: int = 300,
) -> EvalReport:
    """Build the aggregated eval report.

    When a WESAD dataset is present under `wesad_root`, the deviation sections carry
    REAL stress-vs-baseline numbers (dataset="WESAD", via T8.1 HR + T8.2 adapter) and
    the WESAD gap is dropped; otherwise they run on synthetic smoke and the gap is
    logged. Likewise a present `ppg_root` adds a real `ppg_hr` (PPG-DaLiA) section
    (classical DSP HR, plus the encoder when `ppg_encoder_checkpoint` is given). Pass
    `wesad_root=None` / `ppg_root=None` to force the synthetic path (deterministic tests).
    """
    vs = versions or VersionRegistry.from_env().current()
    wesad_wired = wesad_root is not None and wesad_available(wesad_root)
    if wesad_wired:
        assert wesad_root is not None
        labelled = load_wesad_labelled_deviations(wesad_root, max_subjects=wesad_max_subjects)
        detection, calibration = _deviation_sections("WESAD", labelled)
    else:
        detection, calibration = _synthetic_deviation_sections()
    report = EvalReport(
        versions=vs.as_dict(),
        generated_at=(now or datetime.now(UTC)).isoformat(),
    )
    report.sections = [
        _synthetic_retrieval_section(),
        _synthetic_forecast_section(),
        detection,
        calibration,
        _safety_section(),
    ]
    ppg_wired = ppg_root is not None and ppg_dalia_available(ppg_root)
    if ppg_wired:
        assert ppg_root is not None
        report.sections.append(
            _ppg_dalia_hr_section(
                ppg_root,
                encoder_checkpoint=ppg_encoder_checkpoint,
                max_windows_per_subject=ppg_max_windows_per_subject,
            )
        )
    report.gaps = _gaps(wesad_wired=wesad_wired, ppg_wired=ppg_wired)
    return report
