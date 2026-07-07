"""PapageiFeatureExtractor — fine-tuned PaPaGei-S `FeatureExtractor` (docs/16 Sprint 10).

A DEFERRED implementation of the SAME stable `FeatureExtractor` protocol the classical
`WaveformFeatureExtractor` and the from-scratch `FoundationEncoderFeatureExtractor`
implement (docs/02 §6; ai/interfaces) — a new implementation, never a new call site
(CLAUDE.md). Given a raw PPG `SignalWindow` at PaPaGei-S's pretrained rate (125 Hz), it
runs the fine-tuned ResNet trunk (NumPy inference, no torch) and returns
`heart_rate_bpm` (and, when a stress head is configured, `stress_probability`).

Fail-safe by construction (mirrors the from-scratch extractor): a missing checkpoint, a
non-PPG waveform, a sample rate that isn't the model's, or a too-short window all delegate
to the classical DSP extractor. Callers normalise/resample to 125 Hz upstream (the eval
loaders do this via `target_fs_hz`); a mismatched rate deliberately falls back rather than
silently feeding the encoder out-of-distribution input. The raw signal never leaves the
extractor; only the derived features do (CLAUDE.md principle 2).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from schemas.features import FeatureSet, SignalWindow, WaveformKind

from .waveform_extractor import WaveformFeatureExtractor

PAPAGEI_EXTRACTOR_VERSION = "papagei-s-finetuned-v1"


class PapageiFeatureExtractor:
    """Fine-tuned PaPaGei-S PPG→HR extractor with a classical fallback."""

    def __init__(
        self,
        weights: object | None,
        *,
        version: str = PAPAGEI_EXTRACTOR_VERSION,
        fallback: WaveformFeatureExtractor | None = None,
        stress_head: object | None = None,
    ) -> None:
        self._weights = weights  # PapageiEncoderWeights | None (None => always fall back)
        self._version = version
        self._fallback = fallback or WaveformFeatureExtractor()
        self._stress_head = stress_head  # StressHead | None (adds stress_probability)

    @property
    def version(self) -> str:
        """The learned extractor's identity — stamped on every FeatureSet it emits."""
        return self._version

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        version: str = PAPAGEI_EXTRACTOR_VERSION,
        fallback: WaveformFeatureExtractor | None = None,
        stress_head_path: Path | None = None,
    ) -> PapageiFeatureExtractor:
        """Load a fine-tuned PaPaGei-S encoder; a missing/broken checkpoint yields a
        fallback-only extractor (never raises — a missing model must not break the
        pipeline). An optional stress head adds `stress_probability` from the same
        embedding; a missing/broken one is silently skipped.

        On a successful load the emitted version folds in the checkpoint's
        content-addressed id (`{version}+{name@hash}`) so two different PaPaGei
        artifacts are distinguishable in the audit trail — a bare constant would make
        them look identical. A fallback-only extractor keeps the plain base version.
        The load catch is deliberately broad (corrupt manifest/NPZ, bad shapes, missing
        arrays all reduce to fallback): a broken model must degrade, never crash."""
        weights: object | None = None
        stamped = version
        _CKPT_LOAD_ERRORS = (OSError, ValueError, KeyError, zipfile.BadZipFile)
        try:
            from ai.training.checkpoints import load_papagei_weights

            weights = load_papagei_weights(path)
            stamped = f"{version}+{path.name}"  # content-addressed ckpt id (docs/04 §7)
        except _CKPT_LOAD_ERRORS:
            weights = None
        stress_head: object | None = None
        if stress_head_path is not None:
            try:
                from ai.training.checkpoints import load_stress_head

                stress_head = load_stress_head(stress_head_path)
            except _CKPT_LOAD_ERRORS:
                stress_head = None
        return cls(weights, version=stamped, fallback=fallback, stress_head=stress_head)

    def _can_encode(self, window: SignalWindow) -> bool:
        if self._weights is None or window.waveform is None:
            return False
        wf = window.waveform
        from ai.training.papagei_resnet import PapageiEncoderWeights

        assert isinstance(self._weights, PapageiEncoderWeights)
        return (
            wf.kind is WaveformKind.PPG
            and abs(wf.sample_rate_hz - self._weights.sample_rate_hz) < 1e-6
            and len(wf.samples) >= self._weights.window_samples
        )

    def extract(self, window: SignalWindow) -> FeatureSet:
        if not self._can_encode(window):
            return self._fallback.extract(window)

        from ai.training.papagei_resnet import (
            PapageiEncoderWeights,
            papagei_embedding,
        )

        assert isinstance(self._weights, PapageiEncoderWeights)
        w = self._weights
        samples = np.asarray(window.waveform.samples, dtype=np.float64)  # type: ignore[union-attr]
        segment = samples[-w.window_samples :]  # model's native window length
        # One embedding, shared by every head (HR now; stress-context when configured).
        emb = papagei_embedding(w, segment[np.newaxis, :])
        hr = float((emb @ w.head_w + w.head_b)[0] * w.hr_std + w.hr_mean)
        features: dict[str, float] = {"heart_rate_bpm": hr, "n_samples": float(samples.size)}
        if self._stress_head is not None:
            from ai.training.stress_head import StressHead, predict_stress_proba

            assert isinstance(self._stress_head, StressHead)
            features["stress_probability"] = float(predict_stress_proba(self._stress_head, emb)[0])
        return FeatureSet(
            patient_id=window.patient_id,
            metric_code=window.metric_code,
            context=window.context,
            window_start=window.window_start,
            window_end=window.window_end,
            n_total=int(samples.size),
            n_quality_passing=int(samples.size),
            sqi_threshold_applied=None,  # raw-waveform SQI is UNSET clinical config
            features=features,
            feature_extractor_version=self._version,
        )
