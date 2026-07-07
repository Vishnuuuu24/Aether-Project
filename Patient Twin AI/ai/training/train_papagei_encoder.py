"""Fine-tune PaPaGei-S on PPG-DaLiA HR, then export the NumPy serving encoder (docs/16
Sprint 10 — pretrained-encoder init; closes the DoD's "load pretrained weights" gap).

    python -m ai.training.train_papagei_encoder                 # full run (all subjects)
    python -m ai.training.train_papagei_encoder --max-subjects 3 --epochs 3   # smoke

Loads PPG-DaLiA BVP resampled to PaPaGei-S's 125 Hz / 10 s contract, splits
SUBJECT-held-out (no leakage), FULL-fine-tunes the pretrained trunk + a fresh HR head on
MPS while streaming per-epoch progress, exports a torch-free NumPy encoder, runs a
post-fine-tune parity check (torch == NumPy to machine precision, float64), then writes a
versioned checkpoint. Reported MAE is the NumPy serving path — the production number.

Promotion is NOT automatic (CLAUDE.md principle 5): prints scores + an advisory
recommendation; a human runs the gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ai.eval_datasets.ppg_dalia import SignalWindows, load_ppg_dalia_hr_papagei_windows
from ai.training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    register_checkpoint_version,
    write_papagei_checkpoint,
)
from ai.training.papagei_resnet import predict_hr
from ai.training.promotion import Bar, evaluate_promotion, write_promotion_recommendation
from ai.training.splits import subject_held_out_split
from ai.training.torch_papagei import PapageiFineTuneConfig, fine_tune_papagei

DEFAULT_PPG_DALIA_ROOT = Path("datasets/PPG-DaLiA")
DEFAULT_PAPAGEI_CKPT = Path("models/cache/papagei/papagei_s.pt")
ENCODER_NAME = "papagei-s-hr-encoder"
# The prior from-scratch encoder's held-out MAE (docs/17) — the internal number to beat.
FROM_SCRATCH_MAE = 6.50


def _linear_baseline_mae(
    sig_tr: np.ndarray, y_tr: np.ndarray, sig_va: np.ndarray, y_va: np.ndarray
) -> float:
    """5-stat linear HR head on the SAME split — the Sprint 9 trivial bar."""
    def stats(sig: np.ndarray) -> np.ndarray:
        return np.stack(
            [sig.mean(1), sig.std(1), sig.min(1), sig.max(1), np.ptp(sig, axis=1)], axis=1
        )

    xtr, xva = stats(sig_tr), stats(sig_va)
    mean, std = xtr.mean(0), xtr.std(0)
    std[std == 0] = 1.0
    ztr = np.hstack([(xtr - mean) / std, np.ones((len(xtr), 1))])
    zva = np.hstack([(xva - mean) / std, np.ones((len(xva), 1))])
    coef, *_ = np.linalg.lstsq(ztr, y_tr, rcond=None)
    return float(np.mean(np.abs(zva @ coef - y_va)))


def _bar(value: float, lo: float, hi: float, width: int = 24) -> str:
    frac = 1.0 - min(1.0, max(0.0, (value - lo) / max(1e-6, hi - lo)))
    return "█" * int(round(frac * width)) + "·" * (width - int(round(frac * width)))


def _make_progress(baseline_mae: float) -> object:
    lo, hi = max(1.0, baseline_mae - 8.0), baseline_mae + 4.0

    def on_epoch(log: object) -> None:
        mark = " ★best" if log.is_best else (" ✓" if log.val_mae_bpm < baseline_mae else "")  # type: ignore[attr-defined]
        print(
            f"  epoch {log.epoch:>3}/{log.epochs_total}  "  # type: ignore[attr-defined]
            f"lr {log.learning_rate:.2e}  "  # type: ignore[attr-defined]
            f"train_mse {log.train_mse:5.3f}  "  # type: ignore[attr-defined]
            f"val_MAE {log.val_mae_bpm:6.2f} bpm  "  # type: ignore[attr-defined]
            f"[{_bar(log.val_mae_bpm, lo, hi)}]{mark} ({log.seconds:.0f}s)",  # type: ignore[attr-defined]
            flush=True,
        )

    return on_epoch


def run(
    windows: SignalWindows,
    *,
    config: PapageiFineTuneConfig,
    pretrained_checkpoint: Path,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> int:
    train_idx, val_idx, train_subj, val_subj = subject_held_out_split(
        windows.subject_ids, val_fraction=config.val_fraction, seed=config.seed
    )
    sig_tr, y_tr = windows.signals[train_idx], windows.targets[train_idx]
    sig_va, y_va = windows.signals[val_idx], windows.targets[val_idx]

    print(f"subjects: train={list(train_subj)}  held-out={list(val_subj)}")
    print(f"windows : train={len(y_tr)}  val={len(y_va)}  "
          f"({windows.window_samples} samples @ {windows.sample_rate_hz:.0f} Hz)")
    baseline_mae = _linear_baseline_mae(sig_tr, y_tr, sig_va, y_va)
    print(f"linear-baseline held-out HR MAE: {baseline_mae:.2f} bpm")
    print(f"from-scratch encoder (docs/17): {FROM_SCRATCH_MAE:.2f} bpm — the number to beat")
    print(f"fine-tuning PaPaGei-S trunk (PyTorch/MPS) — {config.epochs} epochs:")

    weights, history = fine_tune_papagei(
        sig_tr, y_tr, sig_va, y_va,
        sample_rate_hz=windows.sample_rate_hz,
        window_samples=windows.window_samples,
        pretrained_checkpoint=str(pretrained_checkpoint),
        config=config,
        on_epoch=_make_progress(baseline_mae),
    )

    val_pred = predict_hr(weights, sig_va)  # NumPy serving path (production number)
    err = val_pred - y_va
    enc_mae, enc_rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

    from ai.training.ppg_hr_eval import classical_hr_predictions

    dsp = classical_hr_predictions(sig_va, windows.sample_rate_hz)
    dsp_mask = np.isfinite(dsp)
    dsp_mae = (
        float(np.mean(np.abs(dsp[dsp_mask] - y_va[dsp_mask]))) if dsp_mask.any() else float("nan")
    )

    provenance: dict[str, object] = {
        "dataset": "PPG-DaLiA (resampled 125 Hz / 10 s)",
        "target": "ground-truth HR (bpm)",
        "init": "PaPaGei-S pretrained (Zenodo 10.5281/zenodo.13983110, BSD-3-Clause-Clear)",
        "train_subjects": list(train_subj),
        "held_out_subjects": list(val_subj),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "window_samples": windows.window_samples,
        "sample_rate_hz": windows.sample_rate_hz,
        "architecture": "ResNet1D 18-block (base 32→512) + linear HR head",
        "fine_tune": "full trunk + fresh head (PyTorch/MPS)",
        "split": "subject-held-out",
    }
    metrics = {
        "mae": enc_mae, "rmse": enc_rmse, "dsp_mae": dsp_mae,
        "baseline_mae": baseline_mae, "from_scratch_mae": FROM_SCRATCH_MAE,
        "n": float(len(y_va)), "best_epoch": float(history.best_epoch),
    }
    handle = write_papagei_checkpoint(
        weights, name=ENCODER_NAME, config=config, provenance=provenance,
        metrics=metrics, root=checkpoint_root,
    )
    register_checkpoint_version(handle.version)

    recommendation = evaluate_promotion(
        handle.version,
        [
            Bar("hr_mae_vs_classical_dsp", enc_mae, dsp_mae, higher_is_better=False),
            Bar("hr_mae_vs_from_scratch", enc_mae, FROM_SCRATCH_MAE, higher_is_better=False),
        ],
    )
    write_promotion_recommendation(recommendation, handle.path)

    print(f"\nbest epoch {history.best_epoch}/{config.epochs}  (best-checkpoint kept, not final)")
    print(f"PaPaGei-S fine-tuned  held-out HR MAE {enc_mae:.2f} bpm (RMSE {enc_rmse:.2f})")
    print(f"classical DSP HR MAE  {dsp_mae:.2f} bpm")
    print(f"from-scratch encoder  {FROM_SCRATCH_MAE:.2f} bpm   "
          f"linear baseline {baseline_mae:.2f} bpm")
    rec = "✅ RECOMMENDED for promotion" if recommendation.recommended else "not recommended"
    print(f"promotion: {rec} (advisory; a human runs the gate) — {recommendation.rationale}")
    print(f"checkpoint: {handle.path}")

    print("\n── log entry (paste into docs/17_Training_Log.md, then add judgement) ──")
    print(f"| run | `{handle.version}` |")
    print(f"| held-out | {list(val_subj)} · n={len(y_va)} |")
    print(f"| PaPaGei-S fine-tuned MAE | **{enc_mae:.2f} bpm** (best epoch {history.best_epoch}) |")
    print(f"| from-scratch / DSP / linear | "
          f"{FROM_SCRATCH_MAE:.2f} / {dsp_mae:.2f} / {baseline_mae:.2f} |")
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 10 PaPaGei-S HR fine-tuning.")
    parser.add_argument("--root", default=str(DEFAULT_PPG_DALIA_ROOT))
    parser.add_argument("--papagei", default=str(DEFAULT_PAPAGEI_CKPT),
                        help="pretrained papagei_s.pt path")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="default: ALL subjects (full-quality run)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap windows/subject; default ALL (only for a fast smoke)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args(argv)

    ckpt = Path(args.papagei)
    if not ckpt.exists():
        print(f"pretrained PaPaGei-S checkpoint not found at {ckpt}", file=sys.stderr)
        return 2
    try:
        windows = load_ppg_dalia_hr_papagei_windows(
            Path(args.root), max_subjects=args.max_subjects,
            max_windows_per_subject=args.max_windows,
        )
    except Exception as exc:  # noqa: BLE001 - CLI surface: report and exit non-zero
        print(f"could not load PPG-DaLiA PaPaGei windows: {exc}", file=sys.stderr)
        return 2

    config = PapageiFineTuneConfig(
        epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size, val_fraction=args.val_fraction,
    )
    return run(windows, config=config, pretrained_checkpoint=ckpt)


if __name__ == "__main__":
    raise SystemExit(_main())
