"""PaPaGei-S port tests (docs/16 Sprint 10 — pretrained-encoder init).

Two layers:
  * NumPy-only structural tests (block schedule, forward shapes, downsampling length)
    run unconditionally — no torch, no checkpoint needed.
  * The PARITY GATE (NumPy forward == real torch model to ~1e-10 float64) is
    skip-guarded on torch being installed AND the authentic papagei_s.pt being on
    disk — mirroring the MLX/PPG-DaLiA skip-guards elsewhere.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from ai.training.papagei_resnet import (
    BASE_FILTERS,
    EMBEDDING_DIM,
    N_BLOCK,
    BatchNormParams,
    BlockParams,
    PapageiEncoderWeights,
    block_geometry,
    papagei_embedding,
    papagei_trunk_forward,
    predict_hr,
)

_TORCH = importlib.util.find_spec("torch") is not None
_PAPAGEI_CKPT = Path("models/cache/papagei/papagei_s.pt")
requires_torch = pytest.mark.skipif(not _TORCH, reason="torch not installed (port-only dep)")
requires_ckpt = pytest.mark.skipif(
    not _PAPAGEI_CKPT.exists(), reason="papagei_s.pt not on disk"
)

_SEG = 1250  # 125 Hz × 10 s


# ---- NumPy-only structural tests (no torch) --------------------------------


def test_block_schedule_matches_papagei_s() -> None:
    blocks = block_geometry()
    assert len(blocks) == N_BLOCK
    assert blocks[0].is_first_block and not blocks[0].downsample
    # downsample at every odd block (downsample_gap=2)
    assert [b.downsample for b in blocks] == [i % 2 == 1 for i in range(N_BLOCK)]
    # 9 downsampling blocks 1250 -> 5 by halving
    assert sum(b.downsample for b in blocks) == 9
    # channels double every 4 blocks from 32 to 512; final block emits 512
    assert blocks[0].in_channels == BASE_FILTERS
    assert blocks[-1].out_channels == EMBEDDING_DIM
    # in/out channels are consistent block-to-block (offset zip: last block has no successor)
    for prev, cur in zip(blocks[:-1], blocks[1:], strict=True):
        assert cur.in_channels == prev.out_channels


def _random_weights(*, seed: int = 0) -> PapageiEncoderWeights:
    """Shape-correct random weights — exercises the NumPy forward without torch."""
    rng = np.random.default_rng(seed)

    def bn(c: int) -> BatchNormParams:
        return BatchNormParams(
            gamma=rng.normal(1.0, 0.1, c), beta=rng.normal(0.0, 0.1, c),
            running_mean=rng.normal(0.0, 0.1, c),
            running_var=np.abs(rng.normal(1.0, 0.1, c)) + 0.1,
        )

    schedule = block_geometry()
    blocks = []
    k = 3
    for m in schedule:
        blocks.append(
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
        )
    return PapageiEncoderWeights(
        first_conv_w=rng.normal(0, 0.1, (BASE_FILTERS, 1, k)),
        first_conv_b=rng.normal(0, 0.1, BASE_FILTERS),
        first_bn=bn(BASE_FILTERS),
        blocks=tuple(blocks),
        final_bn=bn(EMBEDDING_DIM),
        head_w=rng.normal(0, 0.1, EMBEDDING_DIM),
        head_b=0.5, hr_mean=70.0, hr_std=10.0,
        sample_rate_hz=125.0, window_samples=_SEG,
    )


def test_numpy_forward_shapes() -> None:
    w = _random_weights()
    sig = np.random.default_rng(1).standard_normal((3, _SEG))
    emb = papagei_embedding(w, sig)
    assert emb.shape == (3, EMBEDDING_DIM)
    hr = predict_hr(w, sig)
    assert hr.shape == (3,)
    assert np.all(np.isfinite(hr))


def test_trunk_forward_is_deterministic() -> None:
    w = _random_weights()
    x = np.random.default_rng(2).standard_normal((2, 1, _SEG))
    a = papagei_trunk_forward(w, x)
    b = papagei_trunk_forward(w, x)
    assert np.array_equal(a, b)


# ---- parity gate (torch reference vs NumPy port) ---------------------------


@requires_torch
@requires_ckpt
def test_numpy_matches_torch_pretrained() -> None:
    """GATE: the NumPy trunk forward reproduces the real PaPaGei-S to ~machine eps."""
    from ai.training.convert_papagei_weights import _parity_check

    max_abs, max_rel = _parity_check(_PAPAGEI_CKPT, n=8, seed=0)
    assert max_abs < 1e-9, f"parity abs error too high: {max_abs:.3e}"
    assert max_rel < 1e-7, f"parity rel error too high: {max_rel:.3e}"


@requires_torch
@requires_ckpt
def test_numpy_matches_torch_various_batch() -> None:
    """Parity holds across batch sizes (no batch-dependent bug)."""
    from ai.training.convert_papagei_weights import _parity_check

    for n in (1, 5):
        max_abs, _ = _parity_check(_PAPAGEI_CKPT, n=n, seed=n)
        assert max_abs < 1e-9
