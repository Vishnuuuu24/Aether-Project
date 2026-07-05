"""Classical HR/HRV-from-waveform DSP (docs/05 §3; CLAUDE.md — "classical features
in v1; PaPaGei-S / Pulse-PPG deferred behind `FeatureExtractor`").

Pure, CPU-only signal processing: it turns a raw ECG or PPG window into a heart
rate (and, where the beat series is clean enough, HRV) by detecting beats and
measuring the intervals between them. No clinical thresholds are applied here —
only physiological *plausibility* bounds (a human heart beats ~30–220 bpm), which
are used to reject detection artefacts, not to make clinical judgements.

  ECG → R-peak detection (Pan–Tompkins-style: band-pass → derivative → square →
        moving-window integration → refractory peak picking).
  PPG → systolic-peak detection (band-pass → refractory peak picking).

`extract_hr_hrv` is the single entry point; `WaveformFeatureExtractor`
(`ai/features/waveform_extractor.py`) plugs it behind the stable `FeatureExtractor`
interface (no new call sites — CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from schemas.features import WaveformKind

# Bare-ndarray aliases: this repo's mypy (1.11) predates numpy 2.x's `NDArray` alias,
# which it rejects as "not a valid type" — `np.ndarray[...]` type-checks cleanly.
FloatArray = np.ndarray[Any, np.dtype[np.float64]]
IntArray = np.ndarray[Any, np.dtype[np.intp]]

# Physiological plausibility bounds on instantaneous heart rate. These are NOT
# clinical thresholds — they bound what a real human heartbeat interval can be, so
# that detector artefacts (double-counted peaks, dropouts) are excluded before any
# statistic is computed. Widely-used DSP defaults, not a clinician-set value.
_MIN_PLAUSIBLE_BPM = 30.0
_MAX_PLAUSIBLE_BPM = 220.0
_MAX_RR_MS = 60_000.0 / _MIN_PLAUSIBLE_BPM  # 2000 ms
_MIN_RR_MS = 60_000.0 / _MAX_PLAUSIBLE_BPM  # ~272 ms


@dataclass(frozen=True)
class HrHrvResult:
    """HR/HRV derived from one waveform window.

    `heart_rate_bpm` is None when too few plausible beats were detected to trust a
    rate (fewer than 2 beats). HRV requires more beats still (`hrv_*` need enough
    consecutive intervals) and is None otherwise — honest absence, never a guess.
    """

    heart_rate_bpm: float | None
    hrv_sdnn_ms: float | None
    hrv_rmssd_ms: float | None
    n_beats: int
    mean_rr_ms: float | None


def _bandpass(signal: FloatArray, fs: float, low: float, high: float) -> FloatArray:
    """Zero-phase Butterworth band-pass. `high` is clamped below Nyquist."""
    nyquist = fs / 2.0
    high = min(high, nyquist * 0.99)
    if low >= high:
        return signal
    b, a = butter(2, [low / nyquist, high / nyquist], btype="band")
    # filtfilt needs len(signal) > 3 * max(len(a), len(b)); guard short windows.
    padlen = 3 * max(len(a), len(b))
    if signal.size <= padlen:
        return signal
    filtered: FloatArray = np.asarray(filtfilt(b, a, signal), dtype=np.float64)
    return filtered


def detect_r_peaks_ecg(signal: FloatArray, fs: float) -> IntArray:
    """Pan–Tompkins-style R-peak detection. Returns sample indices of R-peaks."""
    filtered = _bandpass(signal, fs, 5.0, 15.0)
    derivative = np.diff(filtered, prepend=filtered[:1])
    squared = derivative**2
    # Moving-window integration over ~150 ms (the QRS complex width).
    win = max(1, int(0.150 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    refractory = max(1, int(0.200 * fs))  # 200 ms min RR (<= 300 bpm) — artefact guard
    # Adaptive height: a fraction of the signal's energy, robust to amplitude scale.
    height = float(np.mean(integrated) + 0.5 * np.std(integrated))
    peaks, _ = find_peaks(integrated, distance=refractory, height=height)
    r_peaks: IntArray = np.asarray(peaks, dtype=np.intp)
    return r_peaks


def detect_systolic_peaks_ppg(signal: FloatArray, fs: float) -> IntArray:
    """Systolic-peak detection for PPG/BVP. Returns sample indices of systolic peaks."""
    filtered = _bandpass(signal, fs, 0.5, 8.0)
    refractory = max(1, int(0.300 * fs))  # 300 ms min RR (<= 200 bpm)
    height = float(np.mean(filtered) + 0.3 * np.std(filtered))
    peaks, _ = find_peaks(filtered, distance=refractory, height=height)
    sys_peaks: IntArray = np.asarray(peaks, dtype=np.intp)
    return sys_peaks


def _plausible_rr_ms(peaks: IntArray, fs: float) -> FloatArray:
    """Beat-to-beat intervals (ms), keeping only physiologically plausible ones."""
    if peaks.size < 2:
        return np.empty(0, dtype=np.float64)
    rr = np.diff(peaks).astype(np.float64) / fs * 1000.0
    kept: FloatArray = rr[(rr >= _MIN_RR_MS) & (rr <= _MAX_RR_MS)]
    return kept


def extract_hr_hrv(signal: FloatArray, fs: float, kind: WaveformKind) -> HrHrvResult:
    """Derive HR and (where feasible) HRV from a raw ECG/PPG window.

    HR = 60000 / mean(plausible RR). HRV: SDNN = std of intervals; RMSSD = root mean
    square of successive differences — both need >= 3 consecutive intervals to mean
    anything, so they stay None below that.
    """
    if fs <= 0.0:
        raise ValueError("sample_rate_hz must be > 0")
    sig = np.asarray(signal, dtype=np.float64)
    if sig.size == 0:
        return HrHrvResult(None, None, None, 0, None)

    if kind is WaveformKind.ECG:
        peaks = detect_r_peaks_ecg(sig, fs)
    elif kind is WaveformKind.PPG:
        peaks = detect_systolic_peaks_ppg(sig, fs)
    else:  # pragma: no cover - exhaustive over the enum
        raise ValueError(f"unsupported waveform kind: {kind}")

    rr = _plausible_rr_ms(peaks, fs)
    n_beats = int(rr.size + 1) if rr.size else int(peaks.size)
    if rr.size == 0:
        return HrHrvResult(None, None, None, n_beats, None)

    mean_rr = float(np.mean(rr))
    hr = 60_000.0 / mean_rr
    sdnn = float(np.std(rr, ddof=1)) if rr.size >= 3 else None
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size >= 3 else None
    return HrHrvResult(
        heart_rate_bpm=hr,
        hrv_sdnn_ms=sdnn,
        hrv_rmssd_ms=rmssd,
        n_beats=n_beats,
        mean_rr_ms=mean_rr,
    )
