"""Train the PPG→HR biosignal encoder end-to-end (docs/16 Sprint 10, flagship).

    python -m ai.training.train_ppg_encoder                    # sensible watchable defaults
    python -m ai.training.train_ppg_encoder --epochs 80 --max-subjects 12

Loads raw PPG-DaLiA BVP windows, splits SUBJECT-held-out (no leakage), trains the
conv encoder on Metal while streaming per-epoch progress, then writes a versioned
checkpoint + a standalone HTML report and prints where to see it. A 5-stat linear
head is fit on the SAME split as the honest baseline the encoder must beat.

Promotion is NOT automatic — this produces the artifact + score; a human runs the
gate and decides (CLAUDE.md principle 5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ai.eval_datasets.ppg_dalia import (
    SignalWindows,
    load_ppg_dalia_hr_signal_windows,
)
from ai.training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    register_checkpoint_version,
    write_encoder_checkpoint,
)
from ai.training.encoder_model import predict_hr
from ai.training.mlx_encoder import EncoderConfig, EpochLog, train_encoder
from ai.training.promotion import Bar, evaluate_promotion, write_promotion_recommendation
from ai.training.report import render_encoder_report, write_report
from ai.training.splits import subject_held_out_split

DEFAULT_PPG_DALIA_ROOT = Path("datasets/PPG-DaLiA")
ENCODER_NAME = "ppg-hr-conv-encoder"


def _linear_baseline_mae(
    sig_tr: np.ndarray, y_tr: np.ndarray, sig_va: np.ndarray, y_va: np.ndarray
) -> float:
    """Fit the 5-stat linear HR head on the SAME split — the bar to beat (Sprint 9)."""
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
    frac = 1.0 - min(1.0, max(0.0, (value - lo) / max(1e-6, hi - lo)))  # lower MAE = fuller
    filled = int(round(frac * width))
    return "█" * filled + "·" * (width - filled)


def _make_progress(baseline_mae: float) -> object:
    lo, hi = max(1.0, baseline_mae - 8.0), baseline_mae + 4.0

    def on_epoch(log: EpochLog) -> None:
        mark = " ★best" if log.is_best else (" ✓" if log.val_mae_bpm < baseline_mae else "")
        print(
            f"  epoch {log.epoch:>3}/{log.epochs_total}  "
            f"lr {log.learning_rate:.2e}  "
            f"train_mse {log.train_mse:5.3f}  "
            f"val_MAE {log.val_mae_bpm:6.2f} bpm  "
            f"[{_bar(log.val_mae_bpm, lo, hi)}]{mark}",
            flush=True,
        )

    return on_epoch


def _print_log_stub(
    *,
    version: str,
    provenance: dict[str, object],
    best_epoch: int,
    enc_mae: float,
    dsp_mae: float,
    baseline_mae: float,
) -> None:
    """Emit a paste-ready docs/17 log entry so every run is captured (docs/17 convention)."""
    held = provenance.get("held_out_subjects")
    print("\n── log entry (paste into docs/17_Training_Log.md, then add judgement) ──")
    print(f"| run | `{version}` |")
    print(f"| held-out | {held} · n={provenance.get('n_val')} |")
    print(f"| encoder MAE | **{enc_mae:.2f} bpm** (best epoch {best_epoch}) |")
    print(f"| classical DSP MAE | {dsp_mae:.2f} bpm |")
    print(f"| linear baseline MAE | {baseline_mae:.2f} bpm |")


def run(
    windows: SignalWindows,
    *,
    config: EncoderConfig,
    do_report: bool,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> int:
    train_idx, val_idx, train_subj, val_subj = subject_held_out_split(
        windows.subject_ids, val_fraction=config.val_fraction, seed=config.seed
    )
    sig_tr, y_tr = windows.signals[train_idx], windows.targets[train_idx]
    sig_va, y_va = windows.signals[val_idx], windows.targets[val_idx]

    print(f"subjects: train={list(train_subj)}  held-out={list(val_subj)}")
    print(f"windows : train={len(y_tr)}  val={len(y_va)}  "
          f"({windows.window_samples} samples/window)")
    baseline_mae = _linear_baseline_mae(sig_tr, y_tr, sig_va, y_va)
    print(f"linear-baseline held-out HR MAE (bar to beat): {baseline_mae:.2f} bpm")
    print(f"training conv encoder on MLX — {config.epochs} epochs:")

    weights, history = train_encoder(
        sig_tr, y_tr, sig_va, y_va,
        sample_rate_hz=windows.sample_rate_hz,
        window_samples=windows.window_samples,
        config=config,
        on_epoch=_make_progress(baseline_mae),
    )

    val_pred = predict_hr(weights, sig_va)  # NumPy inference (production path)
    err = val_pred - y_va
    enc_mae, enc_rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

    # Head-to-head vs the classical DSP pipeline — the DoD's literal bar.
    from ai.training.ppg_hr_eval import classical_hr_predictions

    dsp = classical_hr_predictions(sig_va, windows.sample_rate_hz)
    dsp_mask = np.isfinite(dsp)
    dsp_mae = (
        float(np.mean(np.abs(dsp[dsp_mask] - y_va[dsp_mask]))) if dsp_mask.any() else float("nan")
    )
    dsp_cov = float(dsp_mask.mean())

    provenance: dict[str, object] = {
        "dataset": "PPG-DaLiA",
        "target": "ground-truth HR (bpm)",
        "train_subjects": list(train_subj),
        "held_out_subjects": list(val_subj),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "window_samples": windows.window_samples,
        "sample_rate_hz": windows.sample_rate_hz,
        "architecture": "conv1d(32,64,128,128)+GAP+linear",
        "split": "subject-held-out",
    }
    metrics = {
        "mae": enc_mae, "rmse": enc_rmse,
        "baseline_mae": baseline_mae, "n": float(len(y_va)),
    }
    handle = write_encoder_checkpoint(
        weights, name=ENCODER_NAME, config=config, provenance=provenance,
        metrics=metrics, root=checkpoint_root,
    )
    register_checkpoint_version(handle.version)  # stamp derived registry (no mutation)

    # Advisory promotion recommendation (human-gated; never auto-promotes — CLAUDE.md §5).
    recommendation = evaluate_promotion(
        handle.version,
        [
            Bar("hr_mae_vs_classical_dsp", enc_mae, dsp_mae, higher_is_better=False),
            Bar("hr_mae_vs_linear_baseline", enc_mae, baseline_mae, higher_is_better=False),
        ],
    )
    write_promotion_recommendation(recommendation, handle.path)

    beats_dsp = enc_mae < dsp_mae
    beats_lin = enc_mae < baseline_mae
    verdict = "WINS vs classical DSP + linear" if (beats_dsp and beats_lin) else "review"
    print(f"\nbest epoch {history.best_epoch}/{config.epochs}  (best-checkpoint kept, not final)")
    print(f"encoder  held-out HR MAE {enc_mae:.2f} bpm (RMSE {enc_rmse:.2f})")
    print(f"classical DSP HR MAE {dsp_mae:.2f} bpm  (coverage {dsp_cov * 100:.0f}%)")
    print(f"linear baseline  HR MAE {baseline_mae:.2f} bpm")
    print(f"-> {verdict}")
    rec = "✅ RECOMMENDED for promotion" if recommendation.recommended else "not recommended"
    print(f"promotion: {rec} (advisory; a human runs the gate) — {recommendation.rationale}")
    print(f"checkpoint: {handle.path}")
    _print_log_stub(
        version=handle.version, provenance=provenance, best_epoch=history.best_epoch,
        enc_mae=enc_mae, dsp_mae=dsp_mae, baseline_mae=baseline_mae,
    )
    if do_report:
        report_html = render_encoder_report(
            version=handle.version, provenance=provenance, history=history,
            val_true=y_va.tolist(), val_pred=val_pred.tolist(),
            encoder_mae=enc_mae, encoder_rmse=enc_rmse, baseline_mae=baseline_mae,
        )
        report_path = write_report(report_html, version=handle.version)
        print(f"report    : {report_path}  (open in a browser)")
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 10 PPG→HR encoder training.")
    parser.add_argument("--root", default=str(DEFAULT_PPG_DALIA_ROOT))
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="default: ALL subjects (full-quality run)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap windows per subject; default: ALL (only set for a fast test)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val-fraction", type=float, default=0.25,
                        help="fraction of SUBJECTS held out")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args(argv)

    try:
        windows = load_ppg_dalia_hr_signal_windows(
            Path(args.root), max_subjects=args.max_subjects,
            max_windows_per_subject=args.max_windows,
        )
    except Exception as exc:  # noqa: BLE001 - CLI surface: report and exit non-zero
        print(f"could not load PPG-DaLiA signal windows: {exc}", file=sys.stderr)
        return 2

    config = EncoderConfig(
        epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size, val_fraction=args.val_fraction,
    )
    return run(windows, config=config, do_report=not args.no_report)


if __name__ == "__main__":
    raise SystemExit(_main())
