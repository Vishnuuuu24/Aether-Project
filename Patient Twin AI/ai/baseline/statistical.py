"""StatisticalBaselineEngine — v1 `BaselineEngine` (docs/05 §3-5).

Per `(patient, metric_code, context)` (opportunistically stratified by circadian
bucket): rolling **median** center, **MAD**-based robust sigma, **EWMA** trend.
A baseline is `personalised` only after sufficiency (`min_n`, `min_days`);
otherwise it falls back to a labelled population range, or is UNAVAILABLE.

Invariants (docs/05 §8):
  - Learns ONLY from quality-passing readings (SQI gate) — artefacts never move it.
  - Never labels a baseline `personalised` before sufficiency.
  - Every Baseline / DeviationResult honestly carries `is_population_fallback`.

This engine is pure/in-memory: it holds the accepted readings and recomputes on
demand. Persistence, versioning and audit of the resulting nodes are the Patient
State Engine's job (T1.4).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta
from uuid import UUID

from ai.features.sqi import SqiGate
from schemas.baseline import (
    Baseline,
    BaselineAvailability,
    DeviationMagnitude,
    DeviationResult,
    PopulationRange,
)
from schemas.psg import DeviationDirection
from schemas.reading import MeasurementContext, MetricCode, Reading

from .config import BaselineConfig, circadian_bucket
from .population import PopulationReferenceProvider

BASELINE_ENGINE_VERSION = "statistical-v1"

_GroupKey = tuple[MetricCode, MeasurementContext]


class StatisticalBaselineEngine:
    """Implements the `BaselineEngine` protocol (docs/02 §6). Per-patient scoped."""

    def __init__(
        self,
        *,
        gate: SqiGate,
        config: BaselineConfig | None = None,
        population_provider: PopulationReferenceProvider | None = None,
        patient_id: UUID | None = None,
        age_years: int | None = None,
        sex: str | None = None,
        version: str = BASELINE_ENGINE_VERSION,
    ) -> None:
        self._gate = gate
        self._cfg = config or BaselineConfig()
        self._population = population_provider
        self._patient_id = patient_id
        self._age_years = age_years
        self._sex = sex
        self._version = version
        self._readings: dict[_GroupKey, list[Reading]] = defaultdict(list)

    # -- BaselineEngine interface -------------------------------------------

    def update(self, reading: Reading) -> None:
        self._bind_patient(reading.patient_id)
        if not self._gate.passes(reading):
            return  # docs/05 §8: must not learn from sub-threshold readings
        self._readings[(reading.metric_code, reading.context)].append(reading)

    def score(self, reading: Reading) -> DeviationResult:
        self._bind_patient(reading.patient_id)
        baseline = self._baseline_for_scoring(reading)
        value = _as_scalar(reading.value)

        if baseline.availability == BaselineAvailability.UNAVAILABLE or value is None:
            return self._deviation(
                reading, baseline, z_robust=0.0,
                direction=DeviationDirection.NONE, magnitude=DeviationMagnitude.NORMAL,
                confidence=0.0,
            )

        assert baseline.center is not None and baseline.dispersion_sigma is not None
        sigma = baseline.dispersion_sigma
        z_robust = (value - baseline.center) / sigma if sigma > 0.0 else 0.0
        direction = _direction(value, baseline.center)
        magnitude = self._magnitude(abs(z_robust))
        confidence = self._confidence(reading.sqi, baseline)
        return self._deviation(
            reading, baseline, z_robust=z_robust, direction=direction,
            magnitude=magnitude, confidence=confidence,
        )

    def get_baseline(self, metric_code: str, context: str) -> Baseline:
        """Coarse `(metric, context)` baseline (unstratified). Scoring may use a
        finer circadian bucket internally; this returns the representative view.
        """
        return self._compute_baseline(
            MetricCode(metric_code), MeasurementContext(context), bucket=None
        )

    # -- internals -----------------------------------------------------------

    def _bind_patient(self, patient_id: UUID) -> None:
        if self._patient_id is None:
            self._patient_id = patient_id
        elif patient_id != self._patient_id:
            raise ValueError(
                "StatisticalBaselineEngine is per-patient; reading patient_id does not match"
            )

    def _baseline_for_scoring(self, reading: Reading) -> Baseline:
        """Prefer the circadian-bucketed baseline when the metric is circadian AND
        the bucket itself is personalised; otherwise the coarse baseline.
        """
        if reading.metric_code.value in self._cfg.circadian_metrics:
            bucket = circadian_bucket(reading.timestamp)
            stratified = self._compute_baseline(reading.metric_code, reading.context, bucket=bucket)
            if stratified.availability == BaselineAvailability.PERSONALISED:
                return stratified
        return self._compute_baseline(reading.metric_code, reading.context, bucket=None)

    def _compute_baseline(
        self, metric_code: MetricCode, context: MeasurementContext, *, bucket: str | None
    ) -> Baseline:
        pool = self._readings.get((metric_code, context), [])
        if bucket is not None:
            pool = [r for r in pool if circadian_bucket(r.timestamp) == bucket]

        trailing = _trailing_window(pool, self._cfg.window_days)
        values = [v for v in (_as_scalar(r.value) for r in trailing) if v is not None]
        timestamps = [r.timestamp for r in trailing if _as_scalar(r.value) is not None]
        sample_n = len(values)
        as_of = max(timestamps) if timestamps else None
        span_days = _span_days(timestamps)

        personalised = sample_n >= self._cfg.min_n and span_days >= float(self._cfg.min_days)

        if personalised:
            center = statistics.median(values)
            sigma = self._cfg.mad_scale * _mad(values, center)
            return self._baseline(
                metric_code, context, bucket, BaselineAvailability.PERSONALISED,
                center=center, sigma=sigma, ewma=_ewma(trailing, self._cfg.ewma_half_life_days),
                sample_n=sample_n, span_days=span_days, as_of=as_of,
            )

        fallback = self._population_range(metric_code, context)
        if fallback is not None:
            center = (fallback.low + fallback.high) / 2.0
            # Statistical convention: treat the reference range as ~±2σ (≈95% span),
            # so sigma ≈ (high - low) / 4. The clinical values are low/high (config);
            # this is only how a range maps to a scale — no clinical number invented.
            sigma = (fallback.high - fallback.low) / 4.0
            return self._baseline(
                metric_code, context, bucket, BaselineAvailability.POPULATION_FALLBACK,
                center=center, sigma=sigma, ewma=None,
                sample_n=sample_n, span_days=span_days, as_of=as_of,
            )

        return self._baseline(
            metric_code, context, bucket, BaselineAvailability.UNAVAILABLE,
            center=None, sigma=None, ewma=None,
            sample_n=sample_n, span_days=span_days, as_of=as_of,
        )

    def _population_range(
        self, metric_code: MetricCode, context: MeasurementContext
    ) -> PopulationRange | None:
        if self._population is None:
            return None
        return self._population.range_for(
            metric_code.value, context.value, age_years=self._age_years, sex=self._sex
        )

    def _baseline(
        self, metric_code: MetricCode, context: MeasurementContext, bucket: str | None,
        availability: BaselineAvailability, *, center: float | None, sigma: float | None,
        ewma: float | None, sample_n: int, span_days: float, as_of: datetime | None,
    ) -> Baseline:
        assert self._patient_id is not None
        return Baseline(
            patient_id=self._patient_id,
            metric_code=metric_code,
            context=context,
            availability=availability,
            center=center,
            dispersion_sigma=sigma,
            ewma=ewma,
            sample_n=sample_n,
            span_days=span_days,
            window_days=self._cfg.window_days,
            min_n=self._cfg.min_n,
            min_days=self._cfg.min_days,
            is_population_fallback=availability == BaselineAvailability.POPULATION_FALLBACK,
            circadian_bucket=bucket,
            as_of=as_of,
            baseline_engine_version=self._version,
        )

    def _deviation(
        self, reading: Reading, baseline: Baseline, *, z_robust: float,
        direction: DeviationDirection, magnitude: DeviationMagnitude, confidence: float,
    ) -> DeviationResult:
        return DeviationResult(
            reading_id=reading.reading_id,
            patient_id=reading.patient_id,
            metric_code=reading.metric_code,
            context=reading.context,
            z_robust=z_robust,
            direction=direction,
            magnitude=magnitude,
            confidence=confidence,
            confidence_calibrated=False,  # heuristic in v1 (docs/05 §5; T5.2)
            is_population_fallback=baseline.is_population_fallback,
            baseline_availability=baseline.availability,
        )

    def _magnitude(self, abs_z: float) -> DeviationMagnitude:
        if abs_z < self._cfg.mild_z:
            return DeviationMagnitude.NORMAL
        if abs_z < self._cfg.moderate_z:
            return DeviationMagnitude.MILD
        if abs_z < self._cfg.marked_z:
            return DeviationMagnitude.MODERATE
        return DeviationMagnitude.MARKED

    def _confidence(self, sqi: float, baseline: Baseline) -> float:
        """Uncalibrated heuristic (docs/05 §5): low sqi or low sample_n => low
        confidence; population fallback is penalised. Never shipped as calibrated.
        """
        sample_factor = min(1.0, baseline.sample_n / self._cfg.min_n) if self._cfg.min_n else 1.0
        fallback_factor = (
            0.5 if baseline.availability == BaselineAvailability.POPULATION_FALLBACK else 1.0
        )
        return max(0.0, min(1.0, sqi * sample_factor * fallback_factor))


# -- pure helpers -----------------------------------------------------------


def _as_scalar(value: float | dict[str, object]) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _trailing_window(pool: list[Reading], window_days: int) -> list[Reading]:
    if not pool:
        return []
    as_of = max(r.timestamp for r in pool)
    cutoff = as_of - timedelta(days=window_days)
    return [r for r in pool if r.timestamp >= cutoff]


def _span_days(timestamps: list[datetime]) -> float:
    if len(timestamps) < 2:
        return 0.0
    return (max(timestamps) - min(timestamps)).total_seconds() / 86400.0


def _mad(values: list[float], center: float) -> float:
    return statistics.median([abs(v - center) for v in values])


def _ewma(readings: Iterable[Reading], half_life_days: float) -> float | None:
    ordered = sorted(
        ((r.timestamp, _as_scalar(r.value)) for r in readings), key=lambda pair: pair[0]
    )
    ewma: float | None = None
    prev_ts: datetime | None = None
    for timestamp, value in ordered:
        if value is None:
            continue
        if ewma is None or prev_ts is None:
            ewma, prev_ts = value, timestamp
            continue
        dt_days = (timestamp - prev_ts).total_seconds() / 86400.0
        decay = 0.5 ** (dt_days / half_life_days) if half_life_days > 0 else 0.0
        ewma = decay * ewma + (1.0 - decay) * value
        prev_ts = timestamp
    return ewma


def _direction(value: float, center: float) -> DeviationDirection:
    if value > center:
        return DeviationDirection.UP
    if value < center:
        return DeviationDirection.DOWN
    return DeviationDirection.NONE
