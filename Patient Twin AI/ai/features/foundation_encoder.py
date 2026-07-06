"""FoundationEncoderFeatureExtractor — learned PPG→HR `FeatureExtractor` (docs/16 Sprint 10).

A DEFERRED implementation of the SAME stable `FeatureExtractor` protocol the classical
`WaveformFeatureExtractor` implements (docs/02 §6; ai/interfaces) — a new implementation,
never a new call site (CLAUDE.md). Given a raw PPG `SignalWindow`, it runs the trained
conv encoder (NumPy inference, no MLX) and returns `heart_rate_bpm` in a `FeatureSet`.

Fail-safe by construction (docs/16 Sprint 10 "keep a deterministic fallback"): anything
the encoder was not trained for — a missing checkpoint, a non-PPG waveform, a different
sample rate, a too-short window, or no waveform at all — delegates to the classical
extractor. The raw signal never leaves the extractor; only the derived HR does
(CLAUDE.md principle 2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from schemas.features import FeatureSet, SignalWindow, WaveformKind

from .waveform_extractor import WaveformFeatureExtractor

FEATURE_EXTRACTOR_VERSION = "ppg-hr-conv-encoder-v1"


class FoundationEncoderFeatureExtractor:
    """Learned PPG-HR extractor with a classical fallback."""

    def __init__(
        self,
        weights: object | None,
        *,
        version: str = FEATURE_EXTRACTOR_VERSION,
        fallback: WaveformFeatureExtractor | None = None,
    ) -> None:
        self._weights = weights  # EncoderWeights | None (None => always fall back)
        self._version = version
        self._fallback = fallback or WaveformFeatureExtractor()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        version: str = FEATURE_EXTRACTOR_VERSION,
        fallback: WaveformFeatureExtractor | None = None,
    ) -> FoundationEncoderFeatureExtractor:
        """Load a trained encoder; if the checkpoint is absent, return a fallback-only
        extractor (never raises — a missing model must not break the pipeline)."""
        weights: object | None = None
        try:
            from ai.training.checkpoints import load_encoder_weights

            weights = load_encoder_weights(path)
        except (FileNotFoundError, OSError, KeyError):
            weights = None
        return cls(weights, version=version, fallback=fallback)

    def _can_encode(self, window: SignalWindow) -> bool:
        if self._weights is None or window.waveform is None:
            return False
        wf = window.waveform
        from ai.training.encoder_model import EncoderWeights

        assert isinstance(self._weights, EncoderWeights)
        return (
            wf.kind is WaveformKind.PPG
            and abs(wf.sample_rate_hz - self._weights.sample_rate_hz) < 1e-6
            and len(wf.samples) >= self._weights.window_samples
        )

    def extract(self, window: SignalWindow) -> FeatureSet:
        if not self._can_encode(window):
            return self._fallback.extract(window)

        from ai.training.encoder_model import EncoderWeights, predict_hr

        assert isinstance(self._weights, EncoderWeights)
        samples = np.asarray(window.waveform.samples, dtype=np.float64)  # type: ignore[union-attr]
        segment = samples[-self._weights.window_samples :]  # trained window length
        hr = float(predict_hr(self._weights, segment[np.newaxis, :])[0])
        return FeatureSet(
            patient_id=window.patient_id,
            metric_code=window.metric_code,
            context=window.context,
            window_start=window.window_start,
            window_end=window.window_end,
            n_total=int(samples.size),
            n_quality_passing=int(samples.size),
            sqi_threshold_applied=None,  # raw-waveform SQI is UNSET clinical config
            features={"heart_rate_bpm": hr, "n_samples": float(samples.size)},
            feature_extractor_version=self._version,
        )
