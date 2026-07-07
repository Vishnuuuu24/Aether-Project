"""Convert a PyTorch PaPaGei-S trunk into torch-free `PapageiEncoderWeights` (NumPy).

Used two ways:
  * the parity gate (`tests/test_papagei_resnet.py`) — convert the *pretrained* trunk
    and check the NumPy forward matches torch to ~1e-10;
  * the fine-tune export (Stage D) — convert the *fine-tuned* trunk plus the trained HR
    head into the serving container.

Run standalone to sanity-check the pretrained conversion + parity:
    python -m ai.training.convert_papagei_weights
(prints the achieved max abs/rel embedding error vs the real torch model).

`torch` is imported lazily so this module can be referenced without it installed; only
the functions that touch a live torch model need it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ai.training.papagei_resnet import (
    EMBEDDING_DIM,
    BatchNormParams,
    BlockParams,
    PapageiEncoderWeights,
    block_geometry,
    papagei_trunk_forward,
)

if TYPE_CHECKING:  # avoid importing torch at module load
    from ai.training._papagei_reference_torch import ResNet1DMoE

DEFAULT_PAPAGEI_CHECKPOINT = Path("models/cache/papagei/papagei_s.pt")


def _bn_from_torch(module: Any) -> BatchNormParams:
    return BatchNormParams(
        gamma=module.weight.detach().cpu().double().numpy(),
        beta=module.bias.detach().cpu().double().numpy(),
        running_mean=module.running_mean.detach().cpu().double().numpy(),
        running_var=module.running_var.detach().cpu().double().numpy(),
    )


def _arr(t: Any) -> np.ndarray:
    return t.detach().cpu().double().numpy()


def build_encoder_weights(
    model: ResNet1DMoE,
    *,
    head_w: np.ndarray,
    head_b: float,
    hr_mean: float,
    hr_std: float,
    sample_rate_hz: float,
    window_samples: int,
) -> PapageiEncoderWeights:
    """Extract the trunk from a (pretrained or fine-tuned) torch model + attach a head.

    Cross-checks each block's channel/stride/downsample geometry against the
    independently-computed `block_geometry()` schedule, so a mismatch surfaces loudly
    rather than producing a silently-wrong port.
    """
    schedule = block_geometry()
    blocks: list[BlockParams] = []
    for meta, torch_block in zip(schedule, model.basicblock_list, strict=True):
        if (
            torch_block.in_channels != meta.in_channels
            or torch_block.out_channels != meta.out_channels
            or bool(torch_block.downsample) != meta.downsample
            or torch_block.stride != meta.stride
        ):
            raise ValueError(
                f"block geometry mismatch: torch(in={torch_block.in_channels},"
                f"out={torch_block.out_channels},ds={torch_block.downsample},"
                f"stride={torch_block.stride}) vs schedule({meta.in_channels},"
                f"{meta.out_channels},{meta.downsample},{meta.stride})"
            )
        blocks.append(
            BlockParams(
                is_first_block=meta.is_first_block,
                downsample=meta.downsample,
                stride=meta.stride,
                in_channels=meta.in_channels,
                out_channels=meta.out_channels,
                bn1=_bn_from_torch(torch_block.bn1),
                conv1_w=_arr(torch_block.conv1.conv.weight),
                conv1_b=_arr(torch_block.conv1.conv.bias),
                bn2=_bn_from_torch(torch_block.bn2),
                conv2_w=_arr(torch_block.conv2.conv.weight),
                conv2_b=_arr(torch_block.conv2.conv.bias),
            )
        )
    return PapageiEncoderWeights(
        first_conv_w=_arr(model.first_block_conv.conv.weight),
        first_conv_b=_arr(model.first_block_conv.conv.bias),
        first_bn=_bn_from_torch(model.first_block_bn),
        blocks=tuple(blocks),
        final_bn=_bn_from_torch(model.final_bn),
        head_w=np.asarray(head_w, dtype=np.float64),
        head_b=float(head_b),
        hr_mean=float(hr_mean),
        hr_std=float(hr_std),
        sample_rate_hz=float(sample_rate_hz),
        window_samples=int(window_samples),
    )


def load_pretrained_trunk(
    checkpoint_path: Path = DEFAULT_PAPAGEI_CHECKPOINT,
) -> PapageiEncoderWeights:
    """Load the pretrained PaPaGei-S trunk into a NumPy container with an IDENTITY-ish
    placeholder head (zeros) — for parity testing the trunk before any fine-tuning."""
    from ai.training._papagei_reference_torch import (
        PAPAGEI_SEGMENT_SAMPLES,
        PAPAGEI_TARGET_FS_HZ,
        build_papagei_s,
        load_papagei_s_state_dict,
    )

    model = build_papagei_s()
    model = load_papagei_s_state_dict(model, str(checkpoint_path))
    model.double().eval()
    return build_encoder_weights(
        model,
        head_w=np.zeros(EMBEDDING_DIM, dtype=np.float64),
        head_b=0.0,
        hr_mean=0.0,
        hr_std=1.0,
        sample_rate_hz=PAPAGEI_TARGET_FS_HZ,
        window_samples=PAPAGEI_SEGMENT_SAMPLES,
    )


def _parity_check(checkpoint_path: Path, *, n: int = 8, seed: int = 0) -> tuple[float, float]:
    """Convert pretrained trunk, compare NumPy embedding vs real torch on random input.

    Returns (max_abs, max_rel). Feeds identical z-normed input to both paths."""
    import torch

    from ai.training._papagei_reference_torch import (
        PAPAGEI_SEGMENT_SAMPLES,
        build_papagei_s,
        load_papagei_s_state_dict,
    )
    from ai.training.papagei_resnet import znorm_segments

    model = build_papagei_s()
    model = load_papagei_s_state_dict(model, str(checkpoint_path))
    model.double().eval()
    weights = build_encoder_weights(
        model, head_w=np.zeros(EMBEDDING_DIM), head_b=0.0, hr_mean=0.0, hr_std=1.0,
        sample_rate_hz=125.0, window_samples=PAPAGEI_SEGMENT_SAMPLES,
    )

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, PAPAGEI_SEGMENT_SAMPLES))
    x = znorm_segments(raw)  # [N, L], shared by both paths

    with torch.no_grad():
        emb_torch = model.embedding(
            torch.from_numpy(x[:, np.newaxis, :]).double()
        ).cpu().numpy()
    emb_numpy = papagei_trunk_forward(weights, x[:, np.newaxis, :])

    max_abs = float(np.abs(emb_torch - emb_numpy).max())
    denom = np.maximum(np.abs(emb_torch), 1e-8)
    max_rel = float((np.abs(emb_torch - emb_numpy) / denom).max())
    return max_abs, max_rel


def main(argv: list[str] | None = None) -> int:
    checkpoint = DEFAULT_PAPAGEI_CHECKPOINT
    if not checkpoint.exists():
        print(f"PaPaGei-S checkpoint not found at {checkpoint}", file=sys.stderr)
        return 2
    max_abs, max_rel = _parity_check(checkpoint)
    print("pretrained-trunk parity (NumPy vs torch, float64, 512-d embedding):")
    print(f"  max abs error: {max_abs:.3e}")
    print(f"  max rel error: {max_rel:.3e}")
    ok = max_abs < 1e-6
    print("  ->", "PARITY OK" if ok else "PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
