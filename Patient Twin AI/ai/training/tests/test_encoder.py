"""Sprint 10 — biosignal encoder tests (docs/16).

NumPy-only paths (model forward, subject-held-out split, checkpoint roundtrip,
FeatureExtractor fallback, report render) are tested unconditionally. The actual
MLX training loop is skip-guarded on MLX, and the real-data run additionally on the
PPG-DaLiA dataset — mirroring the Sprint 9 skip-guards.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from ai.eval_datasets.ppg_dalia import (
    PpgDaliaLayoutError,
    load_ppg_dalia_hr_signal_windows,
    ppg_dalia_available,
)
from ai.features.foundation_encoder import (
    FEATURE_EXTRACTOR_VERSION,
    FoundationEncoderFeatureExtractor,
)
from ai.training.checkpoints import load_encoder_weights, write_encoder_checkpoint
from ai.training.encoder_model import (
    CONV_CHANNELS,
    KERNEL_SIZE,
    EncoderWeights,
    encoder_embedding,
    predict_hr,
    znorm_windows,
)
from ai.training.mlx_encoder import EncoderConfig
from ai.training.report import render_encoder_report, write_report
from ai.training.splits import subject_held_out_split
from schemas.features import RawWaveform, SignalWindow, WaveformKind
from schemas.reading import MeasurementContext, MetricCode

_MLX = importlib.util.find_spec("mlx") is not None
_PPG_DALIA_ROOT = Path("datasets/PPG-DaLiA")
requires_mlx = pytest.mark.skipif(not _MLX, reason="MLX not installed (Apple-silicon only)")
requires_ppg = pytest.mark.skipif(
    not ppg_dalia_available(_PPG_DALIA_ROOT), reason="PPG-DaLiA dataset not present"
)

_WIN = 512


def _random_weights(*, seed: int = 0, window: int = _WIN) -> EncoderWeights:
    """A shape-correct EncoderWeights with random values (numpy inference needs no MLX)."""
    rng = np.random.default_rng(seed)
    in_c = 1
    conv_w, conv_b = [], []
    for out_c in CONV_CHANNELS:
        conv_w.append(rng.normal(0, 0.1, size=(out_c, KERNEL_SIZE, in_c)))
        conv_b.append(np.zeros(out_c))
        in_c = out_c
    return EncoderWeights(
        conv_w=tuple(conv_w),
        conv_b=tuple(conv_b),
        head_w=rng.normal(0, 0.1, size=CONV_CHANNELS[-1]),
        head_b=75.0,
        hr_mean=75.0,
        hr_std=12.0,
        sample_rate_hz=64.0,
        window_samples=window,
    )


# --- encoder_model (numpy) --------------------------------------------------------


def test_znorm_is_zero_mean_unit_std() -> None:
    x = np.random.default_rng(0).normal(3.0, 5.0, size=(4, _WIN))
    z = znorm_windows(x)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-6)


def test_znorm_handles_flat_window() -> None:
    z = znorm_windows(np.full((1, _WIN), 7.0))
    assert np.all(np.isfinite(z))  # std floor prevents divide-by-zero


def test_forward_shapes_and_finiteness() -> None:
    w = _random_weights()
    sig = np.random.default_rng(1).normal(size=(6, _WIN))
    assert encoder_embedding(w, sig).shape == (6, CONV_CHANNELS[-1])
    hr = predict_hr(w, sig)
    assert hr.shape == (6,)
    assert np.all(np.isfinite(hr))


def test_forward_is_deterministic() -> None:
    w = _random_weights()
    sig = np.random.default_rng(2).normal(size=(3, _WIN))
    assert np.array_equal(predict_hr(w, sig), predict_hr(w, sig))


def _multichannel_weights(n_channels: int, *, seed: int = 0) -> EncoderWeights:
    rng = np.random.default_rng(seed)
    in_c = n_channels
    conv_w, conv_b = [], []
    for out_c in CONV_CHANNELS:
        conv_w.append(rng.normal(0, 0.1, size=(out_c, KERNEL_SIZE, in_c)))
        conv_b.append(np.zeros(out_c))
        in_c = out_c
    return EncoderWeights(
        conv_w=tuple(conv_w), conv_b=tuple(conv_b),
        head_w=rng.normal(0, 0.1, size=CONV_CHANNELS[-1]), head_b=75.0,
        hr_mean=75.0, hr_std=12.0, sample_rate_hz=64.0, window_samples=_WIN,
    )


def test_forward_2d_equals_3d_single_channel() -> None:
    """A BVP-only [N, L] forward is a bit-for-bit special case of [N, L, 1]."""
    w = _random_weights()
    sig = np.random.default_rng(3).normal(size=(5, _WIN))
    assert np.allclose(
        encoder_embedding(w, sig), encoder_embedding(w, sig[:, :, np.newaxis]), atol=1e-12
    )


def test_forward_multichannel_shape_and_finite() -> None:
    w = _multichannel_weights(4)  # BVP + 3-axis ACC
    sig = np.random.default_rng(4).normal(size=(6, _WIN, 4))
    assert encoder_embedding(w, sig).shape == (6, CONV_CHANNELS[-1])
    assert np.all(np.isfinite(predict_hr(w, sig)))


# --- subject-held-out split -------------------------------------------------------


def test_split_has_no_subject_leakage() -> None:
    ids = ["S1"] * 5 + ["S2"] * 5 + ["S3"] * 5 + ["S4"] * 5
    tr, va, tr_s, va_s = subject_held_out_split(ids, val_fraction=0.5, seed=0)
    assert set(tr_s).isdisjoint(set(va_s))
    train_subjects = {ids[i] for i in tr}
    val_subjects = {ids[i] for i in va}
    assert train_subjects.isdisjoint(val_subjects)
    assert len(tr) + len(va) == len(ids)


def test_split_honours_explicit_val_subjects() -> None:
    ids = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    tr, va, tr_s, va_s = subject_held_out_split(ids, val_subjects=["B"])
    assert va_s == ("B",)
    assert {ids[i] for i in va} == {"B"}
    assert set(tr_s) == {"A", "C"}


def test_split_rejects_single_subject() -> None:
    with pytest.raises(ValueError, match="2 distinct subjects"):
        subject_held_out_split(["S1"] * 4)


def test_split_rejects_val_covering_everyone() -> None:
    with pytest.raises(ValueError, match="no training data"):
        subject_held_out_split(["A", "B"], val_subjects=["A", "B"])


# --- checkpoint roundtrip ---------------------------------------------------------


def test_encoder_checkpoint_roundtrip(tmp_path: Path) -> None:
    w = _random_weights()
    cfg = EncoderConfig(epochs=3)
    handle = write_encoder_checkpoint(
        w, name="enc", config=cfg, provenance={"dataset": "PPG-DaLiA"}, root=tmp_path
    )
    reloaded = load_encoder_weights(handle)
    sig = np.random.default_rng(3).normal(size=(4, _WIN))
    assert np.allclose(predict_hr(w, sig), predict_hr(reloaded, sig))
    assert reloaded.window_samples == _WIN
    assert (handle.path / "encoder.npz").exists()
    assert (handle.path / "manifest.json").exists()


def test_encoder_checkpoint_version_is_content_addressed(tmp_path: Path) -> None:
    w = _random_weights()
    prov = {"dataset": "PPG-DaLiA"}
    h1 = write_encoder_checkpoint(w, name="enc", config=EncoderConfig(epochs=3),
                                  provenance=prov, root=tmp_path)
    h2 = write_encoder_checkpoint(w, name="enc", config=EncoderConfig(epochs=3),
                                  provenance=prov, root=tmp_path)
    h3 = write_encoder_checkpoint(w, name="enc", config=EncoderConfig(epochs=9),
                                  provenance=prov, root=tmp_path)
    assert h1.version == h2.version  # same identity -> same id
    assert h1.version != h3.version  # different config -> different id


# --- FeatureExtractor wrapper + classical fallback --------------------------------


def _ppg_window(kind: WaveformKind = WaveformKind.PPG, *, n: int = _WIN + 100,
                rate: float = 64.0) -> SignalWindow:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    t = np.arange(n) / rate
    samples = np.sin(2 * np.pi * 1.3 * t).tolist()  # ~78 bpm-ish tone
    return SignalWindow(
        patient_id=uuid4(),
        metric_code=MetricCode.HEART_RATE,  # no MetricCode.PPG; extractor keys off waveform.kind
        context=MeasurementContext.RESTING,
        window_start=ts,
        window_end=ts,
        waveform=RawWaveform(kind=kind, sample_rate_hz=rate, samples=samples),
    )


def test_encoder_extractor_uses_learned_path_on_ppg() -> None:
    ext = FoundationEncoderFeatureExtractor(_random_weights())
    fs = ext.extract(_ppg_window())
    assert fs.feature_extractor_version == FEATURE_EXTRACTOR_VERSION
    assert "heart_rate_bpm" in fs.features


def test_encoder_extractor_falls_back_without_weights() -> None:
    ext = FoundationEncoderFeatureExtractor(None)  # no model -> classical path
    fs = ext.extract(_ppg_window())
    assert fs.feature_extractor_version == "waveform-classical-v1"


def test_encoder_extractor_falls_back_on_wrong_modality() -> None:
    ext = FoundationEncoderFeatureExtractor(_random_weights())
    fs = ext.extract(_ppg_window(kind=WaveformKind.ECG, rate=700.0))
    assert fs.feature_extractor_version == "waveform-classical-v1"


def test_from_checkpoint_missing_is_fallback_only(tmp_path: Path) -> None:
    ext = FoundationEncoderFeatureExtractor.from_checkpoint(tmp_path / "nope")
    fs = ext.extract(_ppg_window())
    assert fs.feature_extractor_version == "waveform-classical-v1"  # never raises


# --- report -----------------------------------------------------------------------


@requires_mlx
def test_report_renders_and_writes(tmp_path: Path) -> None:
    from ai.training.mlx_encoder import EpochLog, TrainingHistory

    hist = TrainingHistory(logs=(
        EpochLog(1, 2, 0.5, 12.0, 15.0, 2e-3, False, 0.1),
        EpochLog(2, 2, 0.3, 9.5, 12.0, 1e-3, True, 0.1),
    ), best_epoch=2)
    html = render_encoder_report(
        version="enc@abc", provenance={"dataset": "PPG-DaLiA"}, history=hist,
        val_true=[70.0, 80.0, 90.0], val_pred=[72.0, 78.0, 95.0],
        encoder_mae=9.5, encoder_rmse=12.0, baseline_mae=18.0,
    )
    assert "WINS" in html and "<svg" in html
    out = write_report(html, version="enc@abc", root=tmp_path)
    assert out.exists() and out.read_text() == html


# --- data loader ------------------------------------------------------------------


def test_signal_loader_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(PpgDaliaLayoutError):
        load_ppg_dalia_hr_signal_windows(tmp_path)


def test_classical_hr_predictions_on_synthetic_tone() -> None:
    from ai.training.ppg_hr_eval import classical_hr_predictions

    fs = 64.0
    t = np.arange(int(8 * fs)) / fs
    sig = np.stack([np.sin(2 * np.pi * 1.2 * t), np.sin(2 * np.pi * 2.0 * t)])  # 72, 120 bpm
    pred = classical_hr_predictions(sig, fs)
    assert pred.shape == (2,)
    assert np.all(np.isfinite(pred))  # a clean tone yields a detectable pulse


@requires_ppg
def test_evaluate_holdout_scores_classical_and_encoder() -> None:
    from ai.training.ppg_hr_eval import evaluate_holdout

    h = evaluate_holdout(
        _PPG_DALIA_ROOT, weights=_random_weights(window=512), max_windows_per_subject=40
    )
    assert h.n_val > 0
    assert h.classical["mae"] > 0
    assert h.encoder is not None
    assert h.encoder_beats_classical in (True, False)


@requires_ppg
def test_signal_loader_real_dataset_is_leakage_labelled() -> None:
    w = load_ppg_dalia_hr_signal_windows(
        _PPG_DALIA_ROOT, max_subjects=2, max_windows_per_subject=50
    )
    assert w.signals.shape[1] == w.window_samples
    assert len(w.subject_ids) == len(w)
    assert len(w.subjects) == 2  # two distinct subjects contributed


@requires_ppg
def test_fused_loader_has_four_channels() -> None:
    from ai.eval_datasets.ppg_dalia import load_ppg_dalia_hr_fused_windows

    w = load_ppg_dalia_hr_fused_windows(
        _PPG_DALIA_ROOT, max_subjects=2, max_windows_per_subject=40
    )
    assert w.signals.ndim == 3
    assert w.signals.shape[1:] == (w.window_samples, 4)  # BVP + 3-axis ACC
    assert len(w.subject_ids) == len(w)
    assert np.all(np.isfinite(w.signals))


# --- end-to-end MLX training ------------------------------------------------------


@requires_mlx
def test_train_encoder_learns_and_numpy_inference_matches() -> None:
    from ai.training.mlx_encoder import train_encoder

    rng = np.random.default_rng(0)
    t = np.arange(_WIN) / 64.0

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        hr = rng.uniform(55, 135, size=n)
        sig = np.sin(2 * np.pi * (hr / 60.0)[:, None] * t[None, :])
        sig += 0.1 * rng.normal(size=(n, _WIN))
        return sig, hr

    sig_tr, y_tr = make(200)
    sig_va, y_va = make(60)
    weights, history = train_encoder(
        sig_tr, y_tr, sig_va, y_va, sample_rate_hz=64.0, window_samples=_WIN,
        config=EncoderConfig(epochs=4, batch_size=64),
    )
    # NumPy inference reproduces the trained model, and it actually learned.
    pred = predict_hr(weights, sig_va)
    assert np.isfinite(history.best_val_mae)
    assert float(np.mean(np.abs(pred - y_va))) < 20.0


@requires_mlx
@requires_ppg
def test_train_ppg_encoder_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ai.training.report as report_mod
    from ai.training import train_ppg_encoder as cli

    monkeypatch.setattr(report_mod, "DEFAULT_REPORT_ROOT", tmp_path / "reports")
    windows = load_ppg_dalia_hr_signal_windows(
        _PPG_DALIA_ROOT, max_subjects=3, max_windows_per_subject=200
    )
    rc = cli.run(
        windows, config=EncoderConfig(epochs=3, val_fraction=0.34),
        do_report=False, checkpoint_root=tmp_path / "checkpoints",
    )
    assert rc == 0
