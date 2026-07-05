"""WaveformFeatureExtractor — classical HR/HRV `FeatureExtractor` (docs/05 §3; T8.1).

A new implementation of the stable `FeatureExtractor` protocol (docs/02 §6), NOT a
new call site: given a `SignalWindow` carrying a raw ECG/PPG `waveform`, it derives
heart rate (and HRV where the beat series supports it) via the classical DSP in
`ai.features.waveform` and returns a `FeatureSet` whose `features` map holds the
derived scalars. The deferred foundation-encoder path (PaPaGei-S / Pulse-PPG) will
be yet another implementation of this same protocol (CLAUDE.md).

No clinical interpretation happens here — deviation scoring against *this user's
normal* remains the BaselineEngine's job (docs/05 §5, §8).
"""

from __future__ import annotations

import numpy as np

from schemas.features import FeatureSet, SignalWindow

from .waveform import FloatArray, extract_hr_hrv

FEATURE_EXTRACTOR_VERSION = "waveform-classical-v1"


class WaveformFeatureExtractor:
    """Implements the `FeatureExtractor` protocol for raw-waveform windows."""

    def __init__(self, *, version: str = FEATURE_EXTRACTOR_VERSION) -> None:
        self._version = version

    def extract(self, window: SignalWindow) -> FeatureSet:
        waveform = window.waveform
        if waveform is None:
            raise ValueError(
                "WaveformFeatureExtractor requires a SignalWindow carrying a raw waveform"
            )
        signal: FloatArray = np.asarray(waveform.samples, dtype=np.float64)
        result = extract_hr_hrv(signal, waveform.sample_rate_hz, waveform.kind)

        features: dict[str, float] = {
            "n_samples": float(signal.size),
            "n_beats": float(result.n_beats),
        }
        if result.heart_rate_bpm is not None:
            features["heart_rate_bpm"] = result.heart_rate_bpm
        if result.mean_rr_ms is not None:
            features["mean_rr_ms"] = result.mean_rr_ms
        if result.hrv_sdnn_ms is not None:
            features["hrv_sdnn_ms"] = result.hrv_sdnn_ms
        if result.hrv_rmssd_ms is not None:
            features["hrv_rmssd_ms"] = result.hrv_rmssd_ms

        return FeatureSet(
            patient_id=window.patient_id,
            metric_code=window.metric_code,
            context=window.context,
            window_start=window.window_start,
            window_end=window.window_end,
            n_total=signal.size,
            # No per-sample SQI gate for raw waveforms in v1: every sample is used.
            n_quality_passing=signal.size,
            # Raw-waveform SQI thresholds are UNSET clinical config, not fabricated.
            sqi_threshold_applied=None,
            features=features,
            feature_extractor_version=self._version,
        )
