"""Feature-extraction contracts (docs/02 §6, docs/05 §3-4).

`SignalWindow` is the input to `FeatureExtractor.extract`; `FeatureSet` is its
output. In v1 the extractor operates on reduced `Reading`s (statistical-first).
The DEFERRED foundation-encoder impl extends the same contract with raw-waveform
refs — the call site never changes (docs/05 §3).

No service redefines these; import from here (CLAUDE.md: contracts live in schemas/).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from schemas.reading import MeasurementContext, MetricCode, Reading


class WaveformKind(str, Enum):
    """The kind of raw physiological waveform a window carries (docs/05 §3).

    Selects the classical DSP peak detector: R-peak detection for ECG, systolic-
    peak detection for PPG. The value stream is raw signal *amplitude* — never a
    clinical measurement — so it is deliberately kept off `Reading`.
    """

    ECG = "ecg"
    PPG = "ppg"


class RawWaveform(BaseModel):
    """A raw, uniformly-sampled physiological signal (docs/05 §3).

    This is the raw-waveform extension the deferred foundation-encoder path was
    always specified to consume (see module docstring). In v1 the classical
    `WaveformFeatureExtractor` derives HR/HRV from it. `samples` are raw signal
    amplitudes in device units; the LLM never sees them — only the derived,
    structured `FeatureSet` (CLAUDE.md principle 2).
    """

    kind: WaveformKind
    sample_rate_hz: float = Field(gt=0.0)
    samples: list[float] = Field(min_length=1)


class SignalWindow(BaseModel):
    """A trailing window of readings for one `(patient, metric, context)`.

    The unit over which SQI gating and feature extraction operate (docs/05 §3-4).
    Every reading must share the window's patient_id / metric_code / context.
    A window may instead (or additionally) carry a raw `waveform` — the input to
    the classical DSP `WaveformFeatureExtractor` (T8.1); the reduced-`Reading` and
    raw-`waveform` forms share this one contract so the call site never changes.
    """

    patient_id: UUID
    metric_code: MetricCode
    context: MeasurementContext
    window_start: datetime
    window_end: datetime
    readings: list[Reading] = Field(default_factory=list)
    waveform: RawWaveform | None = None

    @model_validator(mode="after")
    def _window_is_consistent(self) -> SignalWindow:
        for name, moment in (("window_start", self.window_start), ("window_end", self.window_end)):
            if moment.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware — naive datetimes are rejected")
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")
        for reading in self.readings:
            if reading.patient_id != self.patient_id:
                raise ValueError("reading.patient_id does not match the window")
            if reading.metric_code != self.metric_code:
                raise ValueError("reading.metric_code does not match the window")
            if reading.context != self.context:
                raise ValueError("reading.context does not match the window")
        return self


class FeatureSet(BaseModel):
    """Descriptive features over the quality-passing readings in a `SignalWindow`.

    v1 features are purely descriptive statistics — no clinical interpretation
    happens here (deviation scoring is the BaselineEngine's job, docs/05 §5, §8).
    `features` is a metric-agnostic map so the deferred encoder can add richer keys
    without a contract change.
    """

    patient_id: UUID
    metric_code: MetricCode
    context: MeasurementContext
    window_start: datetime
    window_end: datetime
    n_total: int = Field(ge=0)
    n_quality_passing: int = Field(ge=0)
    # None => no clinical SQI threshold configured for this metric (stub state).
    sqi_threshold_applied: float | None = None
    features: dict[str, float] = Field(default_factory=dict)
    feature_extractor_version: str

    @model_validator(mode="after")
    def _counts_consistent(self) -> FeatureSet:
        if self.n_quality_passing > self.n_total:
            raise ValueError("n_quality_passing cannot exceed n_total")
        return self
