"""Accelerometer-fusion experiment: does wrist ACC lower PPG→HR error? (docs/16 Sprint 10).

    python -m ai.training.fusion_experiment                 # full-quality, all subjects
    python -m ai.training.fusion_experiment --epochs 60     # quicker sanity

Motion is what corrupts wrist PPG, so the roadmap's headline "next lever" is fusing the
accelerometer. This trains TWO encoders on the SAME subject-held-out split — one on BVP
only, one on BVP + 3-axis ACC (resampled to the BVP grid) — and reports the held-out HR
MAE of each and the lift. Same recipe for both arms, so the delta is attributable to ACC.

This is a MEASUREMENT (does fusion help, and by how much?). Serving a fused model needs
multi-channel windows in the signal contract, so promotion/serving stays a human decision
informed by this number (CLAUDE.md principle 5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ai.eval_datasets.ppg_dalia import (
    PpgDaliaLayoutError,
    load_ppg_dalia_hr_fused_windows,
    ppg_dalia_available,
)
from ai.training.encoder_model import predict_hr
from ai.training.mlx_encoder import EncoderConfig, train_encoder
from ai.training.splits import subject_held_out_split

DEFAULT_PPG_DALIA_ROOT = Path("datasets/PPG-DaLiA")


def _train_arm(
    name: str, sig_tr: np.ndarray, y_tr: np.ndarray, sig_va: np.ndarray, y_va: np.ndarray,
    *, sample_rate_hz: float, window_samples: int, config: EncoderConfig,
) -> float:
    channels = 1 if sig_tr.ndim == 2 else sig_tr.shape[2]
    print(f"\n[{name}]  channels={channels}  train={len(y_tr)} val={len(y_va)}")

    def on_epoch(log: object) -> None:
        ep = log.epoch  # type: ignore[attr-defined]
        if ep == 1 or ep % 20 == 0 or ep == config.epochs:
            mark = " ★" if log.is_best else ""  # type: ignore[attr-defined]
            print(f"  {name} epoch {ep:>3}/{config.epochs}  "
                  f"val_MAE {log.val_mae_bpm:6.2f} bpm{mark}", flush=True)  # type: ignore[attr-defined]

    weights, history = train_encoder(
        sig_tr, y_tr, sig_va, y_va, sample_rate_hz=sample_rate_hz,
        window_samples=window_samples, config=config, on_epoch=on_epoch,
    )
    mae = float(np.mean(np.abs(predict_hr(weights, sig_va) - y_va)))
    print(f"[{name}]  best epoch {history.best_epoch}  held-out HR MAE {mae:.2f} bpm")
    return mae


def run(root: Path, *, config: EncoderConfig) -> int:
    if not ppg_dalia_available(root):
        print(f"PPG-DaLiA not found under {root} — nothing to run.", file=sys.stderr)
        return 2

    fused = load_ppg_dalia_hr_fused_windows(root)  # signals [N, L, 4]
    tr, va, tr_s, va_s = subject_held_out_split(
        fused.subject_ids, val_fraction=config.val_fraction, seed=config.seed
    )
    print(f"subjects: train={list(tr_s)}  held-out={list(va_s)}")
    y_tr, y_va = fused.targets[tr], fused.targets[va]

    # BVP-only arm = channel 0 of the very same windows (identical split → fair delta).
    bvp_tr, bvp_va = fused.signals[tr][:, :, 0], fused.signals[va][:, :, 0]
    fus_tr, fus_va = fused.signals[tr], fused.signals[va]

    bvp_mae = _train_arm("bvp-only", bvp_tr, y_tr, bvp_va, y_va,
                         sample_rate_hz=fused.sample_rate_hz,
                         window_samples=fused.window_samples, config=config)
    fus_mae = _train_arm("bvp+acc ", fus_tr, y_tr, fus_va, y_va,
                         sample_rate_hz=fused.sample_rate_hz,
                         window_samples=fused.window_samples, config=config)

    lift = bvp_mae - fus_mae
    pct = 100.0 * lift / bvp_mae if bvp_mae else 0.0
    verb = "helps" if lift > 0 else "does not help"
    print(f"\n=== fusion {verb}: BVP {bvp_mae:.2f} → BVP+ACC {fus_mae:.2f} bpm "
          f"(Δ {lift:+.2f}, {pct:+.1f}%) ===")
    print("\n── log entry (paste into docs/17_Training_Log.md, then add judgement) ──")
    print(f"| fusion | held-out {list(va_s)} · n={len(y_va)} · {config.epochs} epochs |")
    print(f"| BVP-only MAE | {bvp_mae:.2f} bpm |")
    print(f"| BVP+ACC MAE | **{fus_mae:.2f} bpm** (Δ {lift:+.2f}) |")
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 10 ACC-fusion experiment.")
    parser.add_argument("--root", default=str(DEFAULT_PPG_DALIA_ROOT))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)
    config = EncoderConfig(epochs=args.epochs, val_fraction=args.val_fraction)
    try:
        return run(Path(args.root), config=config)
    except PpgDaliaLayoutError as exc:
        print(f"PPG-DaLiA layout error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
