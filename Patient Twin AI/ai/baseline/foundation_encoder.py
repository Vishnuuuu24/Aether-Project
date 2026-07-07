"""FoundationEncoderBaselineEngine — learned `BaselineEngine` (docs/16 Sprint 10).

A DEFERRED implementation of the SAME stable `BaselineEngine` protocol the v1
`StatisticalBaselineEngine` implements (docs/02 §6; ai/interfaces) — a new
implementation, never a new call site (CLAUDE.md).

What it adds over the statistical engine: it can ingest a **raw PPG `SignalWindow`**
directly, deriving heart rate through the trained conv encoder
(`FoundationEncoderFeatureExtractor`, NumPy inference) before the deterministic
deviation math runs. The **deviation scoring itself is unchanged** — it delegates to
a `StatisticalBaselineEngine`, so the personal-baseline statistics, sufficiency
gating and fallback-honesty invariants (docs/05 §8) are identical. The learned part
only produces a *better HR value* for a raw window; it never touches the decision.

Fail-safe by construction: a missing checkpoint / non-PPG / wrong-rate / short window
falls back to the classical DSP extractor (inside `FoundationEncoderFeatureExtractor`).
The raw signal never leaves the extractor — only the derived HR reading reaches the
delegate (CLAUDE.md principle 2). Outputs are stamped with this engine's version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import UUID

from ai.baseline.config import BaselineConfig
from ai.baseline.statistical import StatisticalBaselineEngine
from ai.features.foundation_encoder import FoundationEncoderFeatureExtractor
from ai.features.sqi import SqiGate
from schemas.baseline import Baseline, DeviationResult
from schemas.features import FeatureSet, SignalWindow
from schemas.reading import MetricCode, Reading

FOUNDATION_ENCODER_BASELINE_VERSION = "foundation-encoder-baseline-v1"
PAPAGEI_BASELINE_VERSION = "papagei-s-baseline-v1"


@runtime_checkable
class _WindowExtractor(Protocol):
    """Structural type for any learned `FeatureExtractor` this engine can ingest windows
    through — both `FoundationEncoderFeatureExtractor` and `PapageiFeatureExtractor` fit
    (they share `extract()` + a `version` stamp)."""

    version: str

    def extract(self, window: SignalWindow) -> FeatureSet: ...


class FoundationEncoderBaselineEngine:
    """Implements the `BaselineEngine` protocol. Per-patient scoped.

    Compose it from a `FoundationEncoderFeatureExtractor` (learned HR with classical
    fallback) and a `StatisticalBaselineEngine` delegate (the deviation math). The
    `Reading`-based protocol methods delegate straight through; the value-add is the
    `*_from_window` pair that turns a raw PPG window into an HR reading first.
    """

    def __init__(
        self,
        *,
        extractor: _WindowExtractor,
        delegate: StatisticalBaselineEngine,
        version: str = FOUNDATION_ENCODER_BASELINE_VERSION,
    ) -> None:
        self._extractor = extractor
        self._delegate = delegate
        self._version = version

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        gate: SqiGate,
        config: BaselineConfig | None = None,
        patient_id: UUID | None = None,
        age_years: int | None = None,
        sex: str | None = None,
        version: str = FOUNDATION_ENCODER_BASELINE_VERSION,
    ) -> FoundationEncoderBaselineEngine:
        """Build from a trained encoder checkpoint. A missing checkpoint yields a
        fallback-only extractor (never raises) — the engine still works, classically.
        The delegate carries THIS engine's version so outputs are stamped honestly.
        """
        extractor = FoundationEncoderFeatureExtractor.from_checkpoint(path)
        delegate = StatisticalBaselineEngine(
            gate=gate,
            config=config,
            patient_id=patient_id,
            age_years=age_years,
            sex=sex,
            version=version,
        )
        return cls(extractor=extractor, delegate=delegate, version=version)

    @classmethod
    def from_papagei_checkpoint(
        cls,
        path: Path,
        *,
        gate: SqiGate,
        config: BaselineConfig | None = None,
        patient_id: UUID | None = None,
        age_years: int | None = None,
        sex: str | None = None,
        stress_head_path: Path | None = None,
        version: str = PAPAGEI_BASELINE_VERSION,
    ) -> FoundationEncoderBaselineEngine:
        """Same engine, but ingest windows through the fine-tuned PaPaGei-S extractor
        instead of the from-scratch encoder. The deviation math (delegate) is IDENTICAL —
        only the HR-from-window step differs. A missing checkpoint falls back to classical
        inside the extractor (never raises)."""
        from ai.features.papagei_extractor import PapageiFeatureExtractor

        extractor = PapageiFeatureExtractor.from_checkpoint(
            path, stress_head_path=stress_head_path
        )
        delegate = StatisticalBaselineEngine(
            gate=gate,
            config=config,
            patient_id=patient_id,
            age_years=age_years,
            sex=sex,
            version=version,
        )
        return cls(extractor=extractor, delegate=delegate, version=version)

    @property
    def version(self) -> str:
        return self._version

    @property
    def feature_extractor_version(self) -> str:
        return self._extractor.version

    # -- BaselineEngine interface (Reading-based; pure delegation) -----------

    def update(self, reading: Reading) -> None:
        self._delegate.update(reading)

    def score(self, reading: Reading) -> DeviationResult:
        return self._delegate.score(reading)

    def get_baseline(self, metric_code: str, context: str) -> Baseline:
        return self._delegate.get_baseline(metric_code, context)

    # -- learned value-add: ingest a raw PPG window --------------------------

    def update_from_window(self, window: SignalWindow) -> Reading | None:
        """Derive HR from a raw window (encoder or classical fallback) and learn from
        it. Returns the derived reading, or None if no HR could be extracted."""
        reading = self._reading_from_window(window)
        if reading is not None:
            self._delegate.update(reading)
        return reading

    def score_from_window(self, window: SignalWindow) -> DeviationResult | None:
        """Derive HR from a raw window and score it against the personal baseline.
        Returns None if no HR could be extracted (abstention, not a guess)."""
        reading = self._reading_from_window(window)
        if reading is None:
            return None
        return self._delegate.score(reading)

    def _reading_from_window(self, window: SignalWindow) -> Reading | None:
        features = self._extractor.extract(window).features
        hr = features.get("heart_rate_bpm")
        if hr is None:
            return None
        return Reading(
            patient_id=window.patient_id,
            metric_code=MetricCode.HEART_RATE,
            value=float(hr),
            unit="bpm",
            timestamp=window.window_end,
            source_device="foundation_encoder",
            sqi=1.0,  # raw-waveform SQI is UNSET clinical config; accept for scoring
            context=window.context,
            ingest_adapter="foundation_encoder",
        )
