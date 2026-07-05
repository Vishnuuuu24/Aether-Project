"""T8.1: classical HR/HRV-from-waveform DSP, validated on known-answer signals.

A synthetic ECG/PPG waveform is built at an *exact* known heart rate; the detector
must recover that rate within a tight tolerance. This is the "validated against a
known-answer signal" DoD — no clinical thresholds are involved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from ai.features.waveform import (
    detect_r_peaks_ecg,
    detect_systolic_peaks_ppg,
    extract_hr_hrv,
)
from ai.features.waveform_extractor import (
    FEATURE_EXTRACTOR_VERSION,
    WaveformFeatureExtractor,
)
from schemas.features import RawWaveform, SignalWindow, WaveformKind
from schemas.reading import MeasurementContext, MetricCode

_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _synthetic_ecg(bpm: float, fs: float, seconds: float) -> np.ndarray:
    """A crude but detectable ECG: a Gaussian R-peak bump once per beat."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    rr = 60.0 / bpm
    signal = np.zeros(n, dtype=np.float64)
    width = 0.010  # ~10 ms QRS spike
    beat = rr
    while beat < seconds:
        signal += np.exp(-((t - beat) ** 2) / (2 * width**2))
        beat += rr
    # add a little baseline wander + noise to make the filter earn its keep
    signal += 0.05 * np.sin(2 * np.pi * 0.3 * t)
    rng = np.random.default_rng(1)
    signal += 0.01 * rng.standard_normal(n)
    return signal


def _synthetic_ppg(bpm: float, fs: float, seconds: float) -> np.ndarray:
    """A smooth quasi-sinusoidal pulse wave at the given rate."""
    n = int(fs * seconds)
    t = np.arange(n) / fs
    freq = bpm / 60.0
    # asymmetric pulse: fundamental + a weaker second harmonic
    signal = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    return signal.astype(np.float64)


@pytest.mark.parametrize("bpm", [50.0, 72.0, 100.0])
def test_ecg_recovers_known_heart_rate(bpm: float) -> None:
    fs = 250.0
    signal = _synthetic_ecg(bpm, fs, seconds=30.0)
    result = extract_hr_hrv(signal, fs, WaveformKind.ECG)
    assert result.heart_rate_bpm is not None
    assert abs(result.heart_rate_bpm - bpm) < 1.0  # within 1 bpm of ground truth


@pytest.mark.parametrize("bpm", [60.0, 88.0])
def test_ppg_recovers_known_heart_rate(bpm: float) -> None:
    fs = 64.0  # WESAD/Empatica BVP rate
    signal = _synthetic_ppg(bpm, fs, seconds=30.0)
    result = extract_hr_hrv(signal, fs, WaveformKind.PPG)
    assert result.heart_rate_bpm is not None
    assert abs(result.heart_rate_bpm - bpm) < 2.0


def test_r_peak_count_matches_beats() -> None:
    fs = 250.0
    signal = _synthetic_ecg(72.0, fs, seconds=30.0)
    peaks = detect_r_peaks_ecg(signal, fs)
    # 72 bpm over 30 s ≈ 36 beats; detection should be within a beat or two.
    assert 34 <= len(peaks) <= 38


def test_ppg_peak_detection_nonempty() -> None:
    fs = 64.0
    peaks = detect_systolic_peaks_ppg(_synthetic_ppg(60.0, fs, 20.0), fs)
    assert len(peaks) >= 18


def test_constant_signal_yields_no_rate() -> None:
    # A flat line has no beats — HR must be honestly absent, not a fabricated number.
    flat = np.ones(2000, dtype=np.float64)
    result = extract_hr_hrv(flat, 250.0, WaveformKind.ECG)
    assert result.heart_rate_bpm is None
    assert result.n_beats <= 1


def test_hrv_present_for_variable_beats() -> None:
    # Genuinely variable RR intervals should produce non-zero SDNN/RMSSD.
    fs = 250.0
    rng = np.random.default_rng(7)
    t_beats = np.cumsum(rng.uniform(0.75, 0.95, size=40))  # jittered ~72 bpm
    seconds = float(t_beats[-1] + 1.0)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    signal = np.zeros(n, dtype=np.float64)
    for beat in t_beats:
        signal += np.exp(-((t - beat) ** 2) / (2 * 0.010**2))
    result = extract_hr_hrv(signal, fs, WaveformKind.ECG)
    assert result.hrv_sdnn_ms is not None and result.hrv_sdnn_ms > 0.0
    assert result.hrv_rmssd_ms is not None and result.hrv_rmssd_ms > 0.0


def test_extractor_behind_interface_returns_featureset() -> None:
    fs = 250.0
    window = SignalWindow(
        patient_id=uuid4(),
        metric_code=MetricCode.ECG,
        context=MeasurementContext.RESTING,
        window_start=_BASE,
        window_end=_BASE,
        waveform=RawWaveform(
            kind=WaveformKind.ECG,
            sample_rate_hz=fs,
            samples=list(_synthetic_ecg(72.0, fs, 20.0)),
        ),
    )
    fs_out = WaveformFeatureExtractor().extract(window)
    assert fs_out.feature_extractor_version == FEATURE_EXTRACTOR_VERSION
    assert abs(fs_out.features["heart_rate_bpm"] - 72.0) < 1.5
    assert fs_out.features["n_beats"] >= 20
    assert fs_out.sqi_threshold_applied is None  # raw-waveform SQI is UNSET config


def test_extractor_rejects_window_without_waveform() -> None:
    window = SignalWindow(
        patient_id=uuid4(),
        metric_code=MetricCode.ECG,
        context=MeasurementContext.RESTING,
        window_start=_BASE,
        window_end=_BASE,
    )
    with pytest.raises(ValueError, match="requires a SignalWindow carrying a raw waveform"):
        WaveformFeatureExtractor().extract(window)
