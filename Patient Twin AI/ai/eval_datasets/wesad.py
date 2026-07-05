"""WESAD → labelled-deviation adapter (docs/11 §1.2; T8.2).

WESAD (Schmidt et al., ICMI 2018) records chest ECG @ 700 Hz with a per-sample
protocol label. We take the two conditions with a clear physiological contrast —
**baseline** (label 1) vs **stress** (label 2, the TSST) — window the ECG, derive
heart rate per window through the classical `WaveformFeatureExtractor` (T8.1), and
score each window's HR against a personal baseline built from that subject's OWN
baseline-condition windows. Stress windows are the ground-truth positives.

The point is honest, real numbers on the personal-baseline thesis. The layout is
validated before any label is trusted (`WesadLayoutError`); no threshold or label
mapping is invented — the label codes are the dataset's own documented protocol.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ai.baseline.config import BaselineConfig
from ai.baseline.eval import LabelledDeviation
from ai.baseline.statistical import StatisticalBaselineEngine
from ai.features.sqi import SqiGate
from ai.features.waveform import FloatArray, IntArray
from ai.features.waveform_extractor import WaveformFeatureExtractor
from schemas.features import RawWaveform, SignalWindow, WaveformKind
from schemas.reading import MeasurementContext, MetricCode, Reading

# WESAD protocol label codes (dataset-defined; docs/13, datasets/WESAD/README.md).
WESAD_BASELINE_LABEL = 1
WESAD_STRESS_LABEL = 2
_CONDITION_LABELS = (WESAD_BASELINE_LABEL, WESAD_STRESS_LABEL)
# The full documented label alphabet (0=transient, 5/6/7=ignore per protocol).
_VALID_LABELS = frozenset(range(8))
_CHEST_ECG_FS = 700.0  # RespiBAN chest ECG sampling rate (Hz)

_BASE_TS = datetime(2026, 1, 1, tzinfo=UTC)


class WesadLayoutError(ValueError):
    """Raised when a WESAD pickle does not match the documented layout — so we never
    score against mislabelled or misaligned signals."""


def wesad_available(root: Path) -> bool:
    """True when at least one `S*/S*.pkl` subject file is present under `root`."""
    return bool(_subject_files(root))


def _subject_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for sub in sorted(root.glob("S*"), key=lambda p: (len(p.name), p.name)):
        pkl = sub / f"{sub.name}.pkl"
        if pkl.exists():
            out.append(pkl)
    return out


def _load_subject(pkl_path: Path) -> tuple[FloatArray, IntArray]:
    """Load and VALIDATE one subject: returns (chest_ecg, labels), both length-N."""
    with pkl_path.open("rb") as fh:
        raw: Any = pickle.load(fh, encoding="latin1")  # noqa: S301 - trusted local dataset

    if not isinstance(raw, dict) or "signal" not in raw or "label" not in raw:
        raise WesadLayoutError(f"{pkl_path.name}: missing 'signal'/'label' top-level keys")
    signal = raw["signal"]
    if not isinstance(signal, dict) or "chest" not in signal:
        raise WesadLayoutError(f"{pkl_path.name}: signal has no 'chest' block")
    chest = signal["chest"]
    if not isinstance(chest, dict) or "ECG" not in chest:
        raise WesadLayoutError(f"{pkl_path.name}: chest block has no 'ECG'")

    ecg = np.asarray(chest["ECG"], dtype=np.float64).reshape(-1)
    labels = np.asarray(raw["label"]).reshape(-1)
    if ecg.size == 0:
        raise WesadLayoutError(f"{pkl_path.name}: empty ECG")
    if labels.size != ecg.size:
        raise WesadLayoutError(
            f"{pkl_path.name}: label length {labels.size} != ECG length {ecg.size} "
            "(chest signal + label must be co-sampled at 700 Hz)"
        )
    unexpected = set(np.unique(labels).tolist()) - _VALID_LABELS
    if unexpected:
        raise WesadLayoutError(f"{pkl_path.name}: unexpected label codes {sorted(unexpected)}")
    if not {WESAD_BASELINE_LABEL, WESAD_STRESS_LABEL} <= set(np.unique(labels).tolist()):
        raise WesadLayoutError(
            f"{pkl_path.name}: baseline(1) and stress(2) conditions both required"
        )
    return ecg, labels.astype(np.intp)


def _condition_windows(
    ecg: FloatArray,
    labels: IntArray,
    label: int,
    window_samples: int,
) -> list[FloatArray]:
    """Non-overlapping windows fully contained in a single-condition span."""
    idx = np.where(labels == label)[0]
    if idx.size < window_samples:
        return []
    windows: list[FloatArray] = []
    for start in range(int(idx[0]), int(idx[-1]) - window_samples, window_samples):
        end = start + window_samples
        if np.all(labels[start:end] == label):
            windows.append(ecg[start:end])
    return windows


def _hr_reading(
    patient_id: Any, segment: FloatArray, extractor: WaveformFeatureExtractor, ts: datetime
) -> float | None:
    window = SignalWindow(
        patient_id=patient_id,
        metric_code=MetricCode.ECG,
        context=MeasurementContext.RESTING,
        window_start=ts,
        window_end=ts,
        waveform=RawWaveform(
            kind=WaveformKind.ECG, sample_rate_hz=_CHEST_ECG_FS, samples=segment.tolist()
        ),
    )
    features = extractor.extract(window).features
    return features.get("heart_rate_bpm")


def _subject_labelled(
    pkl_path: Path,
    *,
    window_seconds: float,
    extractor: WaveformFeatureExtractor,
    config: BaselineConfig,
) -> list[LabelledDeviation]:
    ecg, labels = _load_subject(pkl_path)
    window_samples = int(window_seconds * _CHEST_ECG_FS)
    patient_id = uuid4()
    # heart_rate derived from raw ECG is accepted wholesale for eval (no per-sample
    # clinical SQI gate exists for raw waveforms; 0.0 = accept, an EVAL choice).
    engine = StatisticalBaselineEngine(
        gate=SqiGate({"heart_rate": 0.0}), config=config, patient_id=patient_id
    )

    def make_reading(hr: float, seq: int) -> Reading:
        return Reading(
            patient_id=patient_id,
            metric_code=MetricCode.HEART_RATE,
            value=hr,
            unit="bpm",
            timestamp=_BASE_TS + timedelta(seconds=seq * window_seconds),
            source_device="wesad_respiban_ecg",
            sqi=1.0,
            context=MeasurementContext.RESTING,
            ingest_adapter="wesad",
        )

    seq = 0
    baseline_readings: list[Reading] = []
    stress_readings: list[Reading] = []
    for label, sink in (
        (WESAD_BASELINE_LABEL, baseline_readings),
        (WESAD_STRESS_LABEL, stress_readings),
    ):
        for segment in _condition_windows(ecg, labels, label, window_samples):
            hr = _hr_reading(patient_id, segment, extractor, _BASE_TS)
            if hr is None:
                continue
            sink.append(make_reading(hr, seq))
            seq += 1

    # Personalise ONLY on baseline-condition HR, then score every window.
    for reading in baseline_readings:
        engine.update(reading)

    labelled: list[LabelledDeviation] = []
    for reading in baseline_readings:
        labelled.append(LabelledDeviation(engine.score(reading), is_abnormal=False))
    for reading in stress_readings:
        labelled.append(LabelledDeviation(engine.score(reading), is_abnormal=True))
    return labelled


def _wesad_config() -> BaselineConfig:
    """Baseline config for a ~2 h single-session recording (not clinical content).

    WESAD spans minutes, not days, so day-scale sufficiency (`min_days`) is relaxed
    to 0 and `min_n` lowered to the number of baseline windows a session yields, and
    circadian stratification is disabled (one sitting → one time-of-day). These are
    EVAL parameters, not the shipped `docs/05` statistical defaults.
    """
    return BaselineConfig(min_n=15, min_days=0, circadian_metrics=frozenset())


def load_wesad_labelled_deviations(
    root: Path,
    *,
    subjects: Sequence[str] | None = None,
    window_seconds: float = 30.0,
    max_subjects: int | None = None,
    extractor: WaveformFeatureExtractor | None = None,
    config: BaselineConfig | None = None,
) -> list[LabelledDeviation]:
    """Parse WESAD subjects into labelled deviations (baseline=normal, stress=abnormal).

    Aggregates across subjects; each subject gets its OWN personal baseline (the
    thesis under test). Raises `WesadLayoutError` if a subject's layout is invalid.
    """
    files = _subject_files(root)
    if subjects is not None:
        wanted = set(subjects)
        files = [p for p in files if p.parent.name in wanted]
    if max_subjects is not None:
        files = files[:max_subjects]
    if not files:
        raise WesadLayoutError(f"no WESAD subject pickles found under {root}")

    ext = extractor or WaveformFeatureExtractor()
    cfg = config or _wesad_config()
    labelled: list[LabelledDeviation] = []
    for pkl_path in files:
        labelled.extend(
            _subject_labelled(pkl_path, window_seconds=window_seconds, extractor=ext, config=cfg)
        )
    return labelled
