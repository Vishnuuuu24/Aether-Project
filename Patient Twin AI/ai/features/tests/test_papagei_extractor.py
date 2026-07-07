"""PapageiFeatureExtractor tests (docs/16 Sprint 10 — pretrained-encoder init).

NumPy-only: no torch, no pretrained checkpoint needed. Random-but-shape-correct
`PapageiEncoderWeights` exercise the encode path; the checkpoint write/load roundtrip and
every fallback branch (missing model, wrong rate, non-PPG, short window) are covered.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from ai.features.papagei_extractor import PAPAGEI_EXTRACTOR_VERSION, PapageiFeatureExtractor
from ai.features.waveform_extractor import (
    FEATURE_EXTRACTOR_VERSION as CLASSICAL_VERSION,
)
from ai.interfaces.feature_extractor import FeatureExtractor
from ai.training.checkpoints import load_papagei_weights, write_papagei_checkpoint
from ai.training.papagei_resnet import (
    BASE_FILTERS,
    EMBEDDING_DIM,
    BatchNormParams,
    BlockParams,
    PapageiEncoderWeights,
    block_geometry,
)
from schemas.features import RawWaveform, SignalWindow, WaveformKind
from schemas.reading import MeasurementContext, MetricCode

_PID = uuid4()
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_FS = 125.0
_WIN = 1250


def _random_weights(*, seed: int = 0) -> PapageiEncoderWeights:
    rng = np.random.default_rng(seed)

    def bn(c: int) -> BatchNormParams:
        return BatchNormParams(
            gamma=rng.normal(1.0, 0.1, c), beta=rng.normal(0.0, 0.1, c),
            running_mean=rng.normal(0.0, 0.1, c),
            running_var=np.abs(rng.normal(1.0, 0.1, c)) + 0.1,
        )

    k = 3
    blocks = [
        BlockParams(
            is_first_block=m.is_first_block, downsample=m.downsample, stride=m.stride,
            in_channels=m.in_channels, out_channels=m.out_channels,
            bn1=bn(m.in_channels),
            conv1_w=rng.normal(0, 0.1, (m.out_channels, m.in_channels, k)),
            conv1_b=rng.normal(0, 0.1, m.out_channels),
            bn2=bn(m.out_channels),
            conv2_w=rng.normal(0, 0.1, (m.out_channels, m.out_channels, k)),
            conv2_b=rng.normal(0, 0.1, m.out_channels),
        )
        for m in block_geometry()
    ]
    return PapageiEncoderWeights(
        first_conv_w=rng.normal(0, 0.1, (BASE_FILTERS, 1, k)),
        first_conv_b=rng.normal(0, 0.1, BASE_FILTERS),
        first_bn=bn(BASE_FILTERS), blocks=tuple(blocks), final_bn=bn(EMBEDDING_DIM),
        head_w=rng.normal(0, 0.1, EMBEDDING_DIM), head_b=0.5,
        hr_mean=70.0, hr_std=10.0, sample_rate_hz=_FS, window_samples=_WIN,
    )


def _ppg_window(
    n: int = _WIN, *, fs: float = _FS, kind: WaveformKind = WaveformKind.PPG
) -> SignalWindow:
    t = np.arange(n) / fs
    samples = np.sin(2 * np.pi * 1.2 * t).tolist()
    return SignalWindow(
        patient_id=_PID, metric_code=MetricCode.HEART_RATE,
        context=MeasurementContext.RESTING, window_start=_BASE, window_end=_BASE,
        waveform=RawWaveform(kind=kind, sample_rate_hz=fs, samples=samples),
    )


def test_is_feature_extractor() -> None:
    assert isinstance(PapageiFeatureExtractor(None), FeatureExtractor)


def test_encodes_ppg_at_native_rate() -> None:
    ext = PapageiFeatureExtractor(_random_weights())
    fs = ext.extract(_ppg_window())
    assert "heart_rate_bpm" in fs.features
    assert np.isfinite(fs.features["heart_rate_bpm"])
    assert fs.feature_extractor_version == PAPAGEI_EXTRACTOR_VERSION


def test_missing_weights_falls_back_to_classical() -> None:
    ext = PapageiFeatureExtractor(None)  # no model
    fs = ext.extract(_ppg_window())
    # classical extractor stamps its OWN version, not the papagei one
    assert fs.feature_extractor_version != PAPAGEI_EXTRACTOR_VERSION
    assert fs.feature_extractor_version == CLASSICAL_VERSION


def test_wrong_rate_falls_back() -> None:
    ext = PapageiFeatureExtractor(_random_weights())
    fs = ext.extract(_ppg_window(fs=64.0))  # not the model's 125 Hz
    assert fs.feature_extractor_version != PAPAGEI_EXTRACTOR_VERSION


def test_non_ppg_falls_back() -> None:
    ext = PapageiFeatureExtractor(_random_weights())
    fs = ext.extract(_ppg_window(kind=WaveformKind.ECG))
    assert fs.feature_extractor_version != PAPAGEI_EXTRACTOR_VERSION


def test_short_window_falls_back() -> None:
    ext = PapageiFeatureExtractor(_random_weights())
    fs = ext.extract(_ppg_window(n=_WIN - 1))
    assert fs.feature_extractor_version != PAPAGEI_EXTRACTOR_VERSION


def test_from_checkpoint_missing_never_raises(tmp_path: Path) -> None:
    ext = PapageiFeatureExtractor.from_checkpoint(tmp_path / "nope")
    fs = ext.extract(_ppg_window())  # falls back, no exception
    assert fs.feature_extractor_version == CLASSICAL_VERSION


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    w = _random_weights(seed=3)
    handle = write_papagei_checkpoint(
        w, name="papagei-test", config={"seed": 3}, provenance={"dataset": "unit"},
        root=tmp_path,
    )
    reloaded = load_papagei_weights(handle.path)
    # forward on the reloaded weights matches the original bit-for-bit
    from ai.training.papagei_resnet import papagei_embedding

    sig = np.random.default_rng(9).standard_normal((2, _WIN))
    assert np.array_equal(papagei_embedding(w, sig), papagei_embedding(reloaded, sig))
    ext = PapageiFeatureExtractor.from_checkpoint(handle.path)
    assert "heart_rate_bpm" in ext.extract(_ppg_window()).features


def test_from_checkpoint_stamps_content_addressed_id(tmp_path: Path) -> None:
    """A loaded checkpoint folds its content-addressed id into the emitted version, so two
    different PaPaGei artifacts are distinguishable in the audit trail (docs/04 §7)."""
    w = _random_weights(seed=5)
    handle = write_papagei_checkpoint(
        w, name="papagei-audit", config={"seed": 5}, provenance={"dataset": "unit"},
        root=tmp_path,
    )
    ext = PapageiFeatureExtractor.from_checkpoint(handle.path)
    stamped = ext.extract(_ppg_window()).feature_extractor_version
    assert stamped.startswith(PAPAGEI_EXTRACTOR_VERSION + "+")
    assert handle.path.name in stamped  # the name@hash checkpoint id


def test_corrupt_checkpoint_falls_back_no_raise(tmp_path: Path) -> None:
    """A corrupt NPZ (not a narrow FileNotFound/OSError) must degrade to the classical
    fallback, never crash the pipeline (fail-safe by construction)."""
    w = _random_weights(seed=7)
    handle = write_papagei_checkpoint(
        w, name="papagei-corrupt", config={"seed": 7}, provenance={"dataset": "unit"},
        root=tmp_path,
    )
    (handle.path / "papagei.npz").write_bytes(b"not a valid npz archive")  # corrupt it
    ext = PapageiFeatureExtractor.from_checkpoint(handle.path)  # must not raise
    fs = ext.extract(_ppg_window())
    assert fs.feature_extractor_version == CLASSICAL_VERSION  # fell back
