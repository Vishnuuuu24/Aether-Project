"""End-to-end smoke job (docs/16 Sprint 9 DoD).

Trains a trivial linear head on a few PPG-DaLiA windows through the shared harness —
data layer → backend → checkpoint → eval hook → version stamp — proving the whole
loop end-to-end WITHOUT depending on any real learned model yet. Later sprints swap
the trivial head for the biosignal encoder; the plumbing exercised here stays.

    python -m ai.training.smoke                 # real PPG-DaLiA (datasets/PPG-DaLiA)
    python -m ai.training.smoke --max-windows 400

The split is a deterministic held-out tail (no shuffle). Sprint 10 tightens this to
subject-held-out splits for the real encoder; at the foundation layer we only prove
the loop runs and is scored.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai.eval_datasets.ppg_dalia import (
    FEATURE_NAMES,
    HrWindows,
    load_ppg_dalia_hr_windows,
)
from ai.training.backends import TrainedHead, select_backend
from ai.training.checkpoints import (
    DEFAULT_CHECKPOINT_ROOT,
    CheckpointHandle,
    register_checkpoint_version,
    write_checkpoint,
)
from ai.training.config import TrainConfig
from ai.training.eval_hook import score_head

DEFAULT_PPG_DALIA_ROOT = Path("datasets/PPG-DaLiA")
SMOKE_NAME = "linear-hr-smoke"


@dataclass(frozen=True)
class SmokeResult:
    version: str
    metrics: dict[str, float]
    n_train: int
    n_val: int
    subjects: tuple[str, ...]
    handle: CheckpointHandle


def _split(windows: HrWindows, val_fraction: float) -> tuple[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]:
    n = len(windows)
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val
    if n_train < 1:
        raise ValueError(f"not enough windows ({n}) to form a train/val split")
    xtr, xva = windows.features[:n_train], windows.features[n_train:]
    ytr, yva = windows.targets[:n_train], windows.targets[n_train:]
    return (xtr, ytr), (xva, yva)


def run_smoke(
    windows: HrWindows,
    *,
    config: TrainConfig | None = None,
    name: str = SMOKE_NAME,
    checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT,
) -> SmokeResult:
    """Train → checkpoint → score → stamp, over a supervised HR-window set."""
    cfg = config or TrainConfig()
    (xtr, ytr), (xva, yva) = _split(windows, cfg.val_fraction)

    backend = select_backend(cfg.backend)
    head = backend.fit(xtr, ytr, cfg)
    # The backend is dataset-agnostic; attach the feature-name provenance here.
    head = TrainedHead(
        weights=head.weights,
        bias=head.bias,
        feature_mean=head.feature_mean,
        feature_std=head.feature_std,
        feature_names=windows.feature_names,
        backend=head.backend,
    )

    metrics = score_head(head, xva, yva)
    provenance: dict[str, object] = {
        "dataset": "PPG-DaLiA",
        "subjects": list(windows.subjects),
        "n_windows": len(windows),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "feature_names": list(windows.feature_names),
    }
    handle = write_checkpoint(
        head,
        name=name,
        config=cfg,
        provenance=provenance,
        metrics=metrics,
        root=checkpoint_root,
    )
    # Stamp the version onto a derived registry (human-gated release; no mutation).
    register_checkpoint_version(handle.version)
    return SmokeResult(
        version=handle.version,
        metrics=metrics,
        n_train=int(len(ytr)),
        n_val=int(len(yva)),
        subjects=windows.subjects,
        handle=handle,
    )


def synthetic_hr_windows(n: int = 300, *, seed: int = 0) -> HrWindows:
    """In-memory PPG-DaLiA-shaped sample (BVP-stat features → HR) for tests that must
    run without the dataset downloaded — mirrors the replay adapter's approach."""
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 1.0, size=(n, len(FEATURE_NAMES)))
    true_w = np.array([8.0, 5.0, -3.0, 2.0, 4.0])
    targets = 75.0 + features @ true_w + rng.normal(0.0, 1.0, size=n)
    return HrWindows(
        features=features, targets=targets, feature_names=FEATURE_NAMES, subjects=("synthetic",)
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sprint 9 training-harness smoke job.")
    parser.add_argument("--root", default=str(DEFAULT_PPG_DALIA_ROOT))
    parser.add_argument("--max-subjects", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=400,
                        help="cap windows per subject for a fast smoke")
    args = parser.parse_args(argv)

    root = Path(args.root)
    try:
        windows = load_ppg_dalia_hr_windows(
            root, max_subjects=args.max_subjects, max_windows_per_subject=args.max_windows
        )
    except Exception as exc:  # noqa: BLE001 - CLI surface: report and exit non-zero
        print(f"could not load PPG-DaLiA windows: {exc}", file=sys.stderr)
        print(f"place subject pickles under {root}/ (see {root}/README.md).", file=sys.stderr)
        return 2

    result = run_smoke(windows)
    print(f"smoke ok  version={result.version}")
    print(f"  subjects={list(result.subjects)}  n_train={result.n_train}  n_val={result.n_val}")
    print(f"  HR MAE={result.metrics['mae']:.2f} bpm  RMSE={result.metrics['rmse']:.2f} bpm")
    print(f"  checkpoint={result.handle.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
