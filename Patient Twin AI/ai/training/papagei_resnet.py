"""Portable NumPy forward for the PaPaGei-S trunk (docs/16 Sprint 10, pretrained init).

PaPaGei-S is fine-tuned in PyTorch (`_papagei_reference_torch.py`, on MPS), but — like
the from-scratch conv encoder (`encoder_model.py`) and the linear `TrainedHead` — it
**serves as plain NumPy** so the feature-extraction / fallback path never needs torch,
MLX, or CUDA. Only fine-tuning does.

This module reproduces the authors' trunk forward (the pre-activation 1-D ResNet up to
the 512-d global-average-pooled embedding) in NumPy, matching PyTorch's channels-first
`[N, C, L]` layout and its SAME-padding convention exactly. It is parity-tested against
the real torch model to ~1e-10 in float64 (`tests/test_papagei_resnet.py`) — that test
is a GATE: a transcription bug here would silently ship a wrong model.

No clinical content lives here — only learned weights and the HR-standardisation stats.
The raw waveform never leaves the extractor; only the derived feature does (principle 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

BN_EPS = 1e-5  # torch.nn.BatchNorm1d default eps
_STD_FLOOR = 1e-6  # guard z-normalisation of a flat window

# PaPaGei-S fixed hyperparameters (README config; mirrored from the torch reference).
BASE_FILTERS = 32
KERNEL_SIZE = 3
STRIDE = 2
N_BLOCK = 18
DOWNSAMPLE_GAP = 2
INCREASEFILTER_GAP = 4
EMBEDDING_DIM = 512


@dataclass(frozen=True)
class BatchNormParams:
    """Eval-mode BatchNorm1d as raw parameters (affine applied in forward)."""

    gamma: FloatArray  # weight [C]
    beta: FloatArray  # bias [C]
    running_mean: FloatArray  # [C]
    running_var: FloatArray  # [C]


@dataclass(frozen=True)
class BlockParams:
    """One pre-activation `BasicBlock`: two conv+BN pairs, plus shortcut geometry."""

    is_first_block: bool
    downsample: bool
    stride: int  # conv1 stride (block stride if downsample else 1)
    in_channels: int
    out_channels: int
    bn1: BatchNormParams
    conv1_w: FloatArray  # [C_out, C_in, K]
    conv1_b: FloatArray  # [C_out]
    bn2: BatchNormParams
    conv2_w: FloatArray  # [C_out, C_out, K]
    conv2_b: FloatArray  # [C_out]


@dataclass(frozen=True)
class PapageiEncoderWeights:
    """All params of the PaPaGei-S trunk + a linear HR head, backend-agnostic (NumPy).

    Mirrors `EncoderWeights` (the from-scratch encoder) in spirit: a serialisable,
    torch-free container the NumPy forward walks. `head_w`/`head_b` regress HR from the
    512-d embedding; `hr_mean`/`hr_std` de-standardise the output.
    """

    first_conv_w: FloatArray  # [BASE_FILTERS, 1, K]
    first_conv_b: FloatArray  # [BASE_FILTERS]
    first_bn: BatchNormParams
    blocks: tuple[BlockParams, ...]
    final_bn: BatchNormParams
    head_w: FloatArray  # [EMBEDDING_DIM]
    head_b: float
    hr_mean: float  # HR target standardisation (de-applied at output)
    hr_std: float
    sample_rate_hz: float  # 125.0
    window_samples: int  # 1250


def block_geometry() -> tuple[BlockParams, ...]:
    """Compute the per-block (is_first, downsample, stride, in/out channels) schedule —
    the SAME loop the torch `ResNet1DMoE.__init__` runs — so ports agree structurally.

    Returns block *metadata* only (weight fields left as empty arrays); the conversion
    script fills the weights. Kept public so the parity test can assert the schedule.
    """
    empty = np.empty(0, dtype=np.float64)
    empty_bn = BatchNormParams(empty, empty, empty, empty)
    blocks: list[BlockParams] = []
    out_channels = BASE_FILTERS
    for i_block in range(N_BLOCK):
        is_first = i_block == 0
        downsample = i_block % DOWNSAMPLE_GAP == 1
        in_channels = (
            BASE_FILTERS
            if is_first
            else int(BASE_FILTERS * 2 ** ((i_block - 1) // INCREASEFILTER_GAP))
        )
        out_channels = (
            in_channels * 2
            if (i_block % INCREASEFILTER_GAP == 0 and i_block != 0)
            else in_channels
        )
        stride = STRIDE if downsample else 1
        blocks.append(
            BlockParams(
                is_first_block=is_first, downsample=downsample, stride=stride,
                in_channels=in_channels, out_channels=out_channels,
                bn1=empty_bn, conv1_w=empty, conv1_b=empty,
                bn2=empty_bn, conv2_w=empty, conv2_b=empty,
            )
        )
    return tuple(blocks)


def _same_pad(length: int, kernel: int, stride: int) -> tuple[int, int]:
    """PaPaGei `MyConv1dPadSame` padding: (left, right) for SAME-style output."""
    out_dim = (length + stride - 1) // stride
    p = max(0, (out_dim - 1) * stride + kernel - length)
    return p // 2, p - p // 2


def _conv1d_same(x: FloatArray, weight: FloatArray, bias: FloatArray, stride: int) -> FloatArray:
    """Conv1d matching torch F.conv1d after SAME pad. x [N, C_in, L], weight
    [C_out, C_in, K], bias [C_out] -> [N, C_out, T]."""
    n, c_in, length = x.shape
    c_out, w_c_in, k = weight.shape
    if w_c_in != c_in:
        raise ValueError(f"conv in-channel mismatch: x has {c_in}, weight expects {w_c_in}")
    pad_left, pad_right = _same_pad(length, k, stride)
    padded = np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)))
    plen = padded.shape[-1]
    t_out = (plen - k) // stride + 1
    last = stride * (t_out - 1) + 1
    out = np.zeros((n, c_out, t_out), dtype=np.float64)
    for j in range(k):
        taps = padded[:, :, j : j + last : stride]  # [N, C_in, T]
        out += np.einsum("nct,oc->not", taps, weight[:, :, j])
    return out + bias[np.newaxis, :, np.newaxis]


def _bn_eval(x: FloatArray, bn: BatchNormParams) -> FloatArray:
    """Eval-mode BatchNorm1d over channels: x [N, C, L]."""
    scale = bn.gamma / np.sqrt(bn.running_var + BN_EPS)
    shift = bn.beta - bn.running_mean * scale
    return x * scale[np.newaxis, :, np.newaxis] + shift[np.newaxis, :, np.newaxis]


def _maxpool_same_downsample(x: FloatArray, stride: int) -> FloatArray:
    """PaPaGei `MyMaxPool1dPadSame(kernel_size=stride)` on the identity shortcut:
    pad SAME with the pool's internal stride=1 (=> pad kernel-1), then MaxPool1d
    with kernel=stride and stride=stride. Zero-padding competes with real values,
    exactly as torch F.pad('constant', 0) does."""
    n, c, length = x.shape
    pad_left, pad_right = _same_pad(length, stride, 1)  # internal stride is 1
    padded = np.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)))
    plen = padded.shape[-1]
    t_out = (plen - stride) // stride + 1
    out = np.empty((n, c, t_out), dtype=np.float64)
    for t in range(t_out):
        s = t * stride
        out[:, :, t] = padded[:, :, s : s + stride].max(axis=-1)
    return out


def _basic_block_forward(x: FloatArray, blk: BlockParams) -> FloatArray:
    """One pre-activation BasicBlock (dropout is eval/no-op, matching torch.eval())."""
    identity = x
    out = x
    if not blk.is_first_block:
        out = _bn_eval(out, blk.bn1)
        out = np.maximum(out, 0.0)  # relu1
    out = _conv1d_same(out, blk.conv1_w, blk.conv1_b, blk.stride)

    out = _bn_eval(out, blk.bn2)
    out = np.maximum(out, 0.0)  # relu2
    out = _conv1d_same(out, blk.conv2_w, blk.conv2_b, 1)

    if blk.downsample:
        identity = _maxpool_same_downsample(identity, blk.stride)
    if blk.out_channels != blk.in_channels:
        ch1 = (blk.out_channels - blk.in_channels) // 2
        ch2 = blk.out_channels - blk.in_channels - ch1
        identity = np.pad(identity, ((0, 0), (ch1, ch2), (0, 0)))
    return out + identity


def papagei_trunk_forward(weights: PapageiEncoderWeights, x: FloatArray) -> FloatArray:
    """Trunk forward on already-normalised input `x` [N, 1, L] -> embedding [N, 512].

    Reproduces the authors' `ResNet1DMoE.embedding` in NumPy. Kept separate from the
    z-norm wrapper so the parity test can feed identical tensors to torch and NumPy."""
    out = _conv1d_same(x, weights.first_conv_w, weights.first_conv_b, 1)
    out = _bn_eval(out, weights.first_bn)
    out = np.maximum(out, 0.0)
    for blk in weights.blocks:
        out = _basic_block_forward(out, blk)
    out = _bn_eval(out, weights.final_bn)
    out = np.maximum(out, 0.0)
    return out.mean(axis=-1)  # global average pool over length -> [N, 512]


def znorm_segments(signals: FloatArray) -> FloatArray:
    """Per-segment z-normalisation (torch_ecg `Normalize`, PaPaGei's preprocessing).

    `signals` is [N, L]; each segment is standardised over its length to mean 0 / std 1.
    """
    x = np.asarray(signals, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = np.maximum(x.std(axis=1, keepdims=True), _STD_FLOOR)
    return (x - mean) / std


def papagei_embedding(weights: PapageiEncoderWeights, signals: FloatArray) -> FloatArray:
    """Serving embedding: raw 125 Hz / 10 s segments [N, L] -> [N, 512].

    Applies the pretrained preprocessing contract (per-segment z-norm) then the trunk.
    """
    x = znorm_segments(signals)[:, np.newaxis, :]  # [N, 1, L]
    return papagei_trunk_forward(weights, x)


def predict_hr(weights: PapageiEncoderWeights, signals: FloatArray) -> FloatArray:
    """Predict heart rate (bpm) for raw 125 Hz / 10 s segments [N, L]."""
    emb = papagei_embedding(weights, signals)
    standardised = emb @ weights.head_w + weights.head_b
    return standardised * weights.hr_std + weights.hr_mean
