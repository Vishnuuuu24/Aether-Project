"""Stress-context head — docs/16 Sprint 10.

Covers the NumPy logistic head (separates classes, needs both classes, checkpoint
roundtrip), the WESAD stress-window loader, and the extractor exposing
`stress_probability` on the same embedding as HR.
"""

from __future__ import annotations

import importlib.util
import pickle
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from ai.eval_datasets.wesad import (
    WESAD_BASELINE_LABEL,
    WESAD_STRESS_LABEL,
    load_wesad_wrist_bvp_stress_windows,
    wesad_available,
)
from ai.features.foundation_encoder import FoundationEncoderFeatureExtractor
from ai.training.checkpoints import load_stress_head, write_stress_head_checkpoint
from ai.training.encoder_model import CONV_CHANNELS, KERNEL_SIZE, EncoderWeights
from ai.training.stress_head import (
    STRESS_HEAD_VERSION,
    StressHead,
    predict_stress_proba,
    train_stress_head,
)
from schemas.features import RawWaveform, SignalWindow, WaveformKind
from schemas.reading import MeasurementContext, MetricCode

_D = CONV_CHANNELS[-1]
_REAL_ROOT = Path("datasets/WESAD")
_FS = 700.0
_WRIST_FS = 64.0
requires_mlx = importlib.util.find_spec("mlx") is not None


def _random_weights(*, seed: int = 0, window: int = 512) -> EncoderWeights:
    rng = np.random.default_rng(seed)
    in_c = 1
    conv_w, conv_b = [], []
    for out_c in CONV_CHANNELS:
        conv_w.append(rng.normal(0, 0.1, size=(out_c, KERNEL_SIZE, in_c)))
        conv_b.append(np.zeros(out_c))
        in_c = out_c
    return EncoderWeights(
        conv_w=tuple(conv_w), conv_b=tuple(conv_b),
        head_w=rng.normal(0, 0.1, size=_D), head_b=75.0,
        hr_mean=75.0, hr_std=12.0, sample_rate_hz=64.0, window_samples=window,
    )


def _two_cluster_embeddings(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=_D)
    a = rng.normal(size=(200, _D)) + 2.0 * direction  # stress
    b = rng.normal(size=(200, _D)) - 2.0 * direction  # baseline
    emb = np.vstack([a, b])
    y = np.concatenate([np.ones(200), np.zeros(200)])
    return emb, y


def test_train_stress_head_separates_classes() -> None:
    emb, y = _two_cluster_embeddings()
    head = train_stress_head(emb, y)
    acc = float(np.mean((predict_stress_proba(head, emb) >= 0.5) == y))
    assert acc > 0.9


def test_train_stress_head_requires_two_classes() -> None:
    emb, _ = _two_cluster_embeddings()
    with pytest.raises(ValueError, match="two windows of each class"):
        train_stress_head(emb, np.ones(emb.shape[0]))


def test_stress_head_checkpoint_roundtrip(tmp_path: Path) -> None:
    emb, y = _two_cluster_embeddings()
    head = train_stress_head(emb, y)
    handle = write_stress_head_checkpoint(
        head, name="t", config={"seed": 0}, provenance={"dataset": "synthetic"}, root=tmp_path
    )
    back = load_stress_head(handle.path)
    assert isinstance(back, StressHead)
    assert back.version == STRESS_HEAD_VERSION
    assert np.allclose(back.w, head.w) and back.b == pytest.approx(head.b)
    assert np.allclose(predict_stress_proba(back, emb), predict_stress_proba(head, emb))


def _stress_head_for_dim() -> StressHead:
    rng = np.random.default_rng(1)
    return StressHead(
        w=rng.normal(size=_D), b=0.0, feat_mean=np.zeros(_D), feat_std=np.ones(_D)
    )


def _ppg_window(n: int = 600) -> SignalWindow:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    t = np.arange(n) / 64.0
    return SignalWindow(
        patient_id=uuid4(), metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING, window_start=ts, window_end=ts,
        waveform=RawWaveform(kind=WaveformKind.PPG, sample_rate_hz=64.0,
                             samples=np.sin(2 * np.pi * 1.3 * t).tolist()),
    )


def test_extractor_emits_stress_probability() -> None:
    ext = FoundationEncoderFeatureExtractor(_random_weights(), stress_head=_stress_head_for_dim())
    fs = ext.extract(_ppg_window())
    assert "heart_rate_bpm" in fs.features
    assert 0.0 <= fs.features["stress_probability"] <= 1.0


def test_extractor_without_stress_head_omits_it() -> None:
    ext = FoundationEncoderFeatureExtractor(_random_weights())
    fs = ext.extract(_ppg_window())
    assert "stress_probability" not in fs.features


def _write_fake_wrist_subject(root: Path, name: str) -> None:
    def bvp(bpm: float, secs: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = int(secs * _WRIST_FS)
        t = np.arange(n) / _WRIST_FS
        sig = np.zeros(n)
        beat = 60.0 / bpm
        while beat < secs:
            sig += np.exp(-((t - beat) ** 2) / (2 * 0.04**2))
            beat += (60.0 / bpm) * float(rng.uniform(0.92, 1.08))
        return sig
    signal = np.concatenate([bvp(65, 120, 1), bvp(100, 80, 2)])
    labels = np.concatenate([
        np.full(int(120 * _FS), WESAD_BASELINE_LABEL, dtype=int),
        np.full(int(80 * _FS), WESAD_STRESS_LABEL, dtype=int),
    ])
    sub = root / name
    sub.mkdir(parents=True)
    with (sub / f"{name}.pkl").open("wb") as fh:
        pickle.dump({"signal": {"wrist": {"BVP": signal.reshape(-1, 1)}}, "label": labels}, fh)


def test_stress_window_loader_synthetic(tmp_path: Path) -> None:
    _write_fake_wrist_subject(tmp_path, "S99")
    w = load_wesad_wrist_bvp_stress_windows(tmp_path, window_seconds=8.0)
    assert len(w) > 0
    assert set(np.unique(w.labels).tolist()) == {0, 1}
    assert w.window_samples == 512
    assert w.subjects == ("S99",)


@pytest.mark.skipif(not wesad_available(_REAL_ROOT), reason="WESAD dataset not on disk")
def test_stress_window_loader_real() -> None:
    w = load_wesad_wrist_bvp_stress_windows(_REAL_ROOT, subjects=["S2"], window_seconds=8.0)
    assert len(w) > 30
    assert set(np.unique(w.labels).tolist()) == {0, 1}
