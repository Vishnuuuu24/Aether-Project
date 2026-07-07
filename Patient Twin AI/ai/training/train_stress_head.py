"""Train + evaluate the PPG stress-context head (docs/16 Sprint 10).

    python -m ai.training.train_stress_head --encoder checkpoints/ppg-hr-conv-encoder@<hash>

Freezes the trained conv encoder, embeds WESAD wrist-BVP windows (baseline vs stress),
and fits a NumPy logistic head on the embedding — SUBJECT-held-out (whole subjects held
out, no leakage). Reports held-out F1 / AUC / accuracy vs a majority-class baseline, then
writes a content-addressed stress-head checkpoint. Promotion stays human-gated.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.eval_datasets.wesad import (
    WesadLayoutError,
    load_wesad_wrist_bvp_stress_windows,
    wesad_available,
)
from ai.training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    load_encoder_weights,
    write_stress_head_checkpoint,
)
from ai.training.promotion import Bar, evaluate_promotion, write_promotion_recommendation
from ai.training.splits import subject_held_out_split
from ai.training.stress_head import (
    STRESS_HEAD_VERSION,
    embed_windows,
    predict_stress_proba,
    train_stress_head,
)

DEFAULT_WESAD_ROOT = Path("datasets/WESAD")
STRESS_HEAD_NAME = "ppg-stress-head"


@dataclass(frozen=True)
class StressScore:
    f1: float
    auc: float
    accuracy: float
    majority_accuracy: float
    n: int
    n_pos: int


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U). Undefined if one class is absent -> nan."""
    pos = p[y > 0.5]
    neg = p[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=np.float64)
    ranks[order] = np.arange(1, p.size + 1, dtype=np.float64)
    # average ties
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    tie_mean = np.zeros(counts.size)
    np.add.at(tie_mean, inv, ranks)
    tie_mean /= counts
    ranks = tie_mean[inv]
    sum_pos = float(ranks[y > 0.5].sum())
    n_pos, n_neg = pos.size, neg.size
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _score(y: np.ndarray, p: np.ndarray, *, threshold: float = 0.5) -> StressScore:
    pred = (p >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = float(np.mean(pred == y))
    n_pos = int(y.sum())
    majority = max(n_pos, y.size - n_pos) / y.size
    return StressScore(
        f1=f1, auc=_auc(y, p), accuracy=accuracy,
        majority_accuracy=majority, n=int(y.size), n_pos=n_pos,
    )


def run(
    *,
    encoder_checkpoint: Path,
    wesad_root: Path,
    seed: int = 0,
    val_fraction: float = 0.25,
    max_subjects: int | None = None,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
    papagei: bool = False,
) -> int:
    if not wesad_available(wesad_root):
        print(f"WESAD not found under {wesad_root} — nothing to train.", file=sys.stderr)
        return 2

    # Same logistic head; only the frozen embedding source differs. PaPaGei-S uses its
    # 125 Hz / 10 s pretrained window contract; the from-scratch encoder uses 64 Hz / 8 s.
    if papagei:
        from ai.training.checkpoints import load_papagei_weights
        from ai.training.papagei_resnet import papagei_embedding

        weights = load_papagei_weights(encoder_checkpoint)
        embed = papagei_embedding
        windows = load_wesad_wrist_bvp_stress_windows(
            wesad_root, max_subjects=max_subjects, target_fs_hz=125.0, window_seconds=10.0
        )
    else:
        weights = load_encoder_weights(encoder_checkpoint)  # type: ignore[assignment]
        embed = embed_windows
        windows = load_wesad_wrist_bvp_stress_windows(wesad_root, max_subjects=max_subjects)
    print(f"windows: {len(windows)}  subjects: {list(windows.subjects)}")
    print(f"stress prevalence: {windows.labels.mean() * 100:.0f}%")

    tr_idx, va_idx, tr_subj, va_subj = subject_held_out_split(
        windows.subject_ids, val_fraction=val_fraction, seed=seed
    )
    print(f"subjects: train={list(tr_subj)}  held-out={list(va_subj)}")

    emb_tr = embed(weights, windows.signals[tr_idx])
    emb_va = embed(weights, windows.signals[va_idx])
    y_tr = windows.labels[tr_idx].astype(np.float64)
    y_va = windows.labels[va_idx].astype(np.float64)

    head = train_stress_head(emb_tr, y_tr, seed=seed)
    p_va = predict_stress_proba(head, emb_va)
    score = _score(y_va, p_va)

    verdict = "beats majority baseline" if score.accuracy > score.majority_accuracy else "review"
    print(f"\nheld-out stress-context: F1 {score.f1:.3f}  AUC {score.auc:.3f}  "
          f"acc {score.accuracy:.3f}  (majority {score.majority_accuracy:.3f}, n={score.n})")
    print(f"-> {verdict}")

    provenance: dict[str, object] = {
        "dataset": "WESAD (wrist BVP)",
        "target": "stress-context (baseline vs TSST)",
        "encoder": "papagei-s-finetuned" if papagei else "from-scratch-conv",
        "encoder_checkpoint": encoder_checkpoint.name,
        "train_subjects": list(tr_subj),
        "held_out_subjects": list(va_subj),
        "n_train": int(y_tr.size),
        "n_val": int(y_va.size),
        "split": "subject-held-out",
        "head": STRESS_HEAD_VERSION,
    }
    metrics = {"f1": score.f1, "auc": score.auc, "accuracy": score.accuracy}
    head_name = "papagei-stress-head" if papagei else STRESS_HEAD_NAME
    handle = write_stress_head_checkpoint(
        head, name=head_name, config={"seed": seed}, provenance=provenance,
        metrics=metrics, root=checkpoint_root,
    )
    recommendation = evaluate_promotion(
        handle.version,
        [Bar("accuracy_vs_majority", score.accuracy, score.majority_accuracy,
             higher_is_better=True)],
    )
    write_promotion_recommendation(recommendation, handle.path)
    rec = "✅ RECOMMENDED" if recommendation.recommended else "not recommended"
    print(f"promotion: {rec} (advisory; human-gated) — {recommendation.rationale}")
    print(f"checkpoint: {handle.path}")
    print("\n── log entry (paste into docs/17_Training_Log.md, then add judgement) ──")
    print(f"| run | `{handle.version}` |")
    print(f"| held-out | {list(va_subj)} · n={score.n} |")
    print(f"| stress F1 / AUC / acc | **{score.f1:.3f}** / {score.auc:.3f} / {score.accuracy:.3f} "
          f"(majority {score.majority_accuracy:.3f}) |")
    return 0


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 10 PPG stress-context head.")
    parser.add_argument("--encoder", required=True, help="encoder checkpoint dir")
    parser.add_argument("--root", default=str(DEFAULT_WESAD_ROOT))
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--papagei", action="store_true",
                        help="the encoder ckpt is a fine-tuned PaPaGei-S (125Hz/10s windows)")
    args = parser.parse_args(argv)
    try:
        return run(
            encoder_checkpoint=Path(args.encoder),
            wesad_root=Path(args.root),
            val_fraction=args.val_fraction,
            max_subjects=args.max_subjects,
            papagei=args.papagei,
        )
    except (WesadLayoutError, FileNotFoundError, ValueError) as exc:
        print(f"stress-head training failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
