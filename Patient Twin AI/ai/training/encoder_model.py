"""Portable 1D-CNN PPG→HR encoder — NumPy forward pass (docs/16 Sprint 10).

The biosignal encoder is trained under MLX (`mlx_encoder.py`) but, like the linear
`TrainedHead` (docs/16 Sprint 9), it runs inference as **plain NumPy** so the
serving path / classical fallback never needs MLX or CUDA — only *training* does.

Architecture (small on purpose, so a run is watchable in minutes and fits the
M5 Pro comfortably): three stride-2 Conv1d+ReLU blocks form the *encoder* trunk,
global-average-pooling yields a 64-d embedding, and a single dense *task head*
regresses heart rate. The encoder embedding is the structured feature that leaves
the extractor — the raw waveform never does (CLAUDE.md principle 2).

Weights are stored in MLX's Conv1d layout `[C_out, kernel, C_in]`; the forward here
reproduces that convolution exactly (parity-tested against MLX). No clinical content
lives here — only learned weights and the target-standardisation stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

# Fixed architecture (kept in sync with the MLX model in mlx_encoder). The NumPy
# forward is generic over depth/width — it just walks `conv_w` — so widening or
# deepening here needs no forward change, only that training builds the same stack.
CONV_CHANNELS: tuple[int, ...] = (32, 64, 128, 128)  # per block; input channels = 1
KERNEL_SIZE = 7
STRIDE = 2
PADDING = 3
EMBEDDING_DIM = CONV_CHANNELS[-1]

_STD_FLOOR = 1e-6  # guard z-normalisation of a flat window


@dataclass(frozen=True)
class EncoderWeights:
    """All parameters of the trained encoder + head, backend-agnostic (NumPy)."""

    conv_w: tuple[FloatArray, ...]  # each [C_out, kernel, C_in]
    conv_b: tuple[FloatArray, ...]  # each [C_out]
    head_w: FloatArray  # [EMBEDDING_DIM]
    head_b: float
    hr_mean: float  # target standardisation (de-applied at output)
    hr_std: float
    sample_rate_hz: float
    window_samples: int


def znorm_windows(signals: FloatArray) -> FloatArray:
    """Per-window z-normalisation: makes HR inference scale-invariant to BVP gain."""
    x = np.asarray(signals, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.maximum(std, _STD_FLOOR)
    return (x - mean) / std


def _conv1d(x: FloatArray, weight: FloatArray, bias: FloatArray) -> FloatArray:
    """Conv1d matching MLX: x [B, L, C_in], weight [C_out, K, C_in] -> [B, T, C_out]."""
    b, length, c_in = x.shape
    c_out, k, w_c_in = weight.shape
    if w_c_in != c_in:
        raise ValueError(f"conv in-channel mismatch: x has {c_in}, weight expects {w_c_in}")
    padded = np.pad(x, ((0, 0), (PADDING, PADDING), (0, 0)))
    t_out = (length + 2 * PADDING - k) // STRIDE + 1
    out = np.zeros((b, t_out, c_out), dtype=np.float64)
    last = STRIDE * (t_out - 1) + 1
    for j in range(k):
        # positions j, j+STRIDE, ... for tap j across all output columns
        taps = padded[:, j : j + last : STRIDE, :]  # [B, T, C_in]
        out += np.einsum("btc,oc->bto", taps, weight[:, j, :])
    return out + bias[np.newaxis, np.newaxis, :]


def encoder_embedding(weights: EncoderWeights, signals: FloatArray) -> FloatArray:
    """Run the conv trunk + GAP, returning the [N, EMBEDDING_DIM] embedding."""
    x = znorm_windows(signals)[:, :, np.newaxis]  # [N, L, 1]
    for w, bias in zip(weights.conv_w, weights.conv_b, strict=True):
        x = np.maximum(_conv1d(x, w, bias), 0.0)  # conv + ReLU
    return x.mean(axis=1)  # global average pool over length -> [N, C_last]


def predict_hr(weights: EncoderWeights, signals: FloatArray) -> FloatArray:
    """Predict heart rate (bpm) for raw BVP windows [N, L]."""
    emb = encoder_embedding(weights, signals)
    standardised = emb @ weights.head_w + weights.head_b
    return standardised * weights.hr_std + weights.hr_mean
