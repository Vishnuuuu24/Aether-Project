"""T8.2: WESAD adapter — layout validation on synthetic pickles (CI-safe) + a
skip-guarded sanity run on the real dataset when it is present on disk."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from ai.baseline.eval import detection_metrics, expected_calibration_error
from ai.eval_datasets.wesad import (
    WESAD_BASELINE_LABEL,
    WESAD_STRESS_LABEL,
    WesadLayoutError,
    load_wesad_labelled_deviations,
    load_wesad_wrist_bvp_labelled_deviations,
    wesad_available,
)
from ai.features.waveform_extractor import WaveformFeatureExtractor

_FS = 700.0
_WRIST_FS = 64.0
_REAL_ROOT = Path("datasets/WESAD")


def _ecg_at(bpm: float, seconds: float, seed: int) -> np.ndarray:
    """A layout-valid ECG at ~`bpm` with realistic beat-to-beat jitter, so the derived
    per-window HR varies (a constant rate would give MAD sigma 0 → no scale)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * _FS)
    t = np.arange(n) / _FS
    sig = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm
    while beat < seconds:
        sig += np.exp(-((t - beat) ** 2) / (2 * 0.010**2))
        beat += (60.0 / bpm) * float(rng.uniform(0.92, 1.08))  # ±8% RR jitter
    return sig


def _write_fake_subject(root: Path, name: str, *, baseline_bpm: float, stress_bpm: float) -> None:
    """A minimal but layout-valid WESAD pickle: baseline then stress span, co-sampled."""
    baseline = _ecg_at(baseline_bpm, 360.0, seed=1)  # 6 min baseline
    stress = _ecg_at(stress_bpm, 240.0, seed=2)  # 4 min stress
    ecg = np.concatenate([baseline, stress])
    labels = np.concatenate(
        [
            np.full(baseline.size, WESAD_BASELINE_LABEL, dtype=int),
            np.full(stress.size, WESAD_STRESS_LABEL, dtype=int),
        ]
    )
    sub = root / name
    sub.mkdir(parents=True)
    with (sub / f"{name}.pkl").open("wb") as fh:
        pickle.dump(
            {"signal": {"chest": {"ECG": ecg.reshape(-1, 1)}}, "label": labels, "subject": name}, fh
        )


def test_available_false_on_empty(tmp_path: Path) -> None:
    assert wesad_available(tmp_path) is False


def test_synthetic_subject_produces_labelled_deviations(tmp_path: Path) -> None:
    _write_fake_subject(tmp_path, "S99", baseline_bpm=65.0, stress_bpm=95.0)
    assert wesad_available(tmp_path) is True
    labelled = load_wesad_labelled_deviations(tmp_path, window_seconds=15.0)
    assert labelled
    # A clean 30-bpm separation must be detected against the personal baseline.
    det = detection_metrics(labelled)
    assert det.recall > 0.8
    _ = expected_calibration_error(labelled)  # must not raise


def test_missing_ecg_raises_layout_error(tmp_path: Path) -> None:
    sub = tmp_path / "S1"
    sub.mkdir()
    with (sub / "S1.pkl").open("wb") as fh:
        pickle.dump({"signal": {"chest": {}}, "label": np.zeros(10)}, fh)
    with pytest.raises(WesadLayoutError, match="chest block has no 'ECG'"):
        load_wesad_labelled_deviations(tmp_path)


def test_label_length_mismatch_raises(tmp_path: Path) -> None:
    sub = tmp_path / "S1"
    sub.mkdir()
    with (sub / "S1.pkl").open("wb") as fh:
        pickle.dump({"signal": {"chest": {"ECG": np.zeros((100, 1))}}, "label": np.zeros(50)}, fh)
    with pytest.raises(WesadLayoutError, match="label length"):
        load_wesad_labelled_deviations(tmp_path)


def test_unexpected_label_code_raises(tmp_path: Path) -> None:
    sub = tmp_path / "S1"
    sub.mkdir()
    labels = np.array([1, 2, 99], dtype=int)
    with (sub / "S1.pkl").open("wb") as fh:
        pickle.dump({"signal": {"chest": {"ECG": np.zeros((3, 1))}}, "label": labels}, fh)
    with pytest.raises(WesadLayoutError, match="unexpected label codes"):
        load_wesad_labelled_deviations(tmp_path)


def test_no_subjects_raises(tmp_path: Path) -> None:
    with pytest.raises(WesadLayoutError, match="no WESAD subject pickles"):
        load_wesad_labelled_deviations(tmp_path)


@pytest.mark.skipif(not wesad_available(_REAL_ROOT), reason="WESAD dataset not on disk")
def test_real_wesad_subject_sanity() -> None:
    labelled = load_wesad_labelled_deviations(_REAL_ROOT, subjects=["S2"])
    assert len(labelled) > 30
    det = detection_metrics(labelled)
    # Honest, non-degenerate real numbers — not asserting a quality bar, just that the
    # harness produces a real, finite result on verified data.
    assert 0.0 <= det.precision <= 1.0
    assert 0.0 <= det.recall <= 1.0
    assert det.tp + det.fn > 0  # some stress windows were actually present


# -- wrist-BVP (PPG @ 64 Hz) path --------------------------------------------


def _bvp_at(bpm: float, seconds: float, seed: int) -> np.ndarray:
    """A layout-valid wrist BVP (PPG) at ~`bpm` with beat-to-beat jitter @ 64 Hz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * _WRIST_FS)
    t = np.arange(n) / _WRIST_FS
    sig = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm
    while beat < seconds:
        sig += np.exp(-((t - beat) ** 2) / (2 * 0.040**2))  # wider systolic pulse
        beat += (60.0 / bpm) * float(rng.uniform(0.92, 1.08))
    return sig


def _write_fake_wrist_subject(
    root: Path, name: str, *, baseline_bpm: float, stress_bpm: float
) -> None:
    """Layout-valid pickle with a wrist BVP block; labels co-sampled at 700 Hz."""
    base_secs, stress_secs = 360.0, 240.0
    bvp = np.concatenate(
        [_bvp_at(baseline_bpm, base_secs, seed=3), _bvp_at(stress_bpm, stress_secs, seed=4)]
    )
    labels = np.concatenate(
        [
            np.full(int(base_secs * _FS), WESAD_BASELINE_LABEL, dtype=int),
            np.full(int(stress_secs * _FS), WESAD_STRESS_LABEL, dtype=int),
        ]
    )
    sub = root / name
    sub.mkdir(parents=True)
    with (sub / f"{name}.pkl").open("wb") as fh:
        pickle.dump({"signal": {"wrist": {"BVP": bvp.reshape(-1, 1)}}, "label": labels}, fh)


def test_synthetic_wrist_bvp_produces_labelled_deviations(tmp_path: Path) -> None:
    _write_fake_wrist_subject(tmp_path, "S99", baseline_bpm=65.0, stress_bpm=100.0)
    labelled = load_wesad_wrist_bvp_labelled_deviations(
        tmp_path, extractor=WaveformFeatureExtractor(), window_seconds=8.0
    )
    assert labelled
    det = detection_metrics(labelled)
    assert det.recall > 0.6  # a clean 35-bpm shift on the wrist channel is detectable
    _ = expected_calibration_error(labelled)  # must not raise


def test_wrist_bvp_missing_block_raises(tmp_path: Path) -> None:
    sub = tmp_path / "S1"
    sub.mkdir()
    labels = np.array([1, 2], dtype=int)
    with (sub / "S1.pkl").open("wb") as fh:
        pickle.dump({"signal": {"chest": {"ECG": np.zeros((2, 1))}}, "label": labels}, fh)
    with pytest.raises(WesadLayoutError, match="no 'wrist'"):
        load_wesad_wrist_bvp_labelled_deviations(tmp_path, extractor=WaveformFeatureExtractor())


def test_wrist_bvp_duration_mismatch_raises(tmp_path: Path) -> None:
    """A truncated/offset wrist stream (BVP span ≠ label span) must fail loudly, not
    silently clip-mislabel its tail (real WESAD is aligned to 0.00 s; guard tolerance 2 s)."""
    sub = tmp_path / "S1"
    sub.mkdir()
    # BVP covers ~300 s @ 64 Hz but the label stream spans ~600 s @ 700 Hz — 300 s drift.
    bvp = np.concatenate([_bvp_at(65.0, 150.0, seed=3), _bvp_at(100.0, 150.0, seed=4)])
    labels = np.concatenate(
        [
            np.full(int(300.0 * _FS), WESAD_BASELINE_LABEL, dtype=int),
            np.full(int(300.0 * _FS), WESAD_STRESS_LABEL, dtype=int),
        ]
    )
    with (sub / "S1.pkl").open("wb") as fh:
        pickle.dump({"signal": {"wrist": {"BVP": bvp.reshape(-1, 1)}}, "label": labels}, fh)
    with pytest.raises(WesadLayoutError, match="differ by"):
        load_wesad_wrist_bvp_labelled_deviations(tmp_path, extractor=WaveformFeatureExtractor())


@pytest.mark.skipif(not wesad_available(_REAL_ROOT), reason="WESAD dataset not on disk")
def test_real_wesad_wrist_bvp_sanity() -> None:
    labelled = load_wesad_wrist_bvp_labelled_deviations(
        _REAL_ROOT, extractor=WaveformFeatureExtractor(), subjects=["S2"], window_seconds=8.0
    )
    assert len(labelled) > 30
    det = detection_metrics(labelled)
    assert 0.0 <= det.precision <= 1.0
    assert det.tp + det.fn > 0
