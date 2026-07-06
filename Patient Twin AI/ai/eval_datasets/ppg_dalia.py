"""PPG-DaLiA → supervised HR-window adapter (docs/16 Sprint 9; T9.1 data layer).

PPG-DaLiA (Reiss et al., "Deep PPG") records wrist Empatica E4 + chest RespiBAN
across 15 subjects, with a **ground-truth heart-rate** label stream computed on 8 s
windows sliding by 2 s. We reuse that documented protocol to cut the wrist BVP
(PPG) into the SAME 8 s windows the labels describe, and pair each window with its
GT HR. Output is a plain `(features, targets)` supervised set that the training
harness (`ai/training/`) consumes for the trivial-linear-head smoke and, later, the
biosignal-encoder heads (Sprint 10).

No clinical content is invented: the target is the dataset's OWN ground-truth HR,
the window geometry is the dataset's OWN protocol, and the features are generic
signal statistics (not thresholds). The layout is validated before any label is
trusted (`PpgDaliaLayoutError`); the sampling-rate constants are *set-with-dataset*
and checked against the array lengths on load.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

# --- Sampling geometry for the PPG-DaLiA release (set-with-dataset; confirmed on load) ---
_BVP_FS_HZ = 64.0  # Empatica E4 photoplethysmography (BVP)
_LABEL_FS_HZ = 0.5  # ground-truth HR: 8 s window shifted by 2 s (matches replay adapter)
_HR_WINDOW_SECONDS = 8.0  # the window each GT-HR label summarises
_HR_STEP_SECONDS = 2.0  # 1 / _LABEL_FS_HZ
_DURATION_TOLERANCE = 0.05  # BVP vs label record duration must agree within 5%

# Generic per-window signal statistics (order is the feature vector layout).
FEATURE_NAMES: tuple[str, ...] = ("mean", "std", "min", "max", "range")


class PpgDaliaLayoutError(ValueError):
    """Raised when a PPG-DaLiA pickle does not match the documented layout — so we
    never train against misaligned signal/label streams."""


@dataclass(frozen=True)
class HrWindows:
    """A supervised HR set: `features` [N, F] paired with GT-HR `targets` [N]."""

    features: FloatArray
    targets: FloatArray
    feature_names: tuple[str, ...]
    subjects: tuple[str, ...]  # provenance: which subject files contributed

    def __len__(self) -> int:
        return int(self.features.shape[0])


def ppg_dalia_available(root: Path) -> bool:
    """True when at least one `S*.pkl` subject file is present under `root`."""
    return bool(_subject_files(root))


def _subject_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("S*.pkl"), key=lambda p: (len(p.name), p.name))


def _load_subject(pkl_path: Path) -> tuple[FloatArray, FloatArray]:
    """Load and VALIDATE one subject: returns (wrist_bvp, gt_hr_labels)."""
    with pkl_path.open("rb") as fh:
        raw: Any = pickle.load(fh, encoding="latin1")  # noqa: S301 - trusted local dataset

    if not isinstance(raw, dict) or "signal" not in raw or "label" not in raw:
        raise PpgDaliaLayoutError(f"{pkl_path.name}: missing 'signal'/'label' top-level keys")
    signal = raw["signal"]
    if not isinstance(signal, dict) or "wrist" not in signal:
        raise PpgDaliaLayoutError(f"{pkl_path.name}: signal has no 'wrist' block")
    wrist = signal["wrist"]
    if not isinstance(wrist, dict) or "BVP" not in wrist:
        raise PpgDaliaLayoutError(f"{pkl_path.name}: wrist block has no 'BVP'")

    bvp = np.asarray(wrist["BVP"], dtype=np.float64).reshape(-1)
    labels = np.asarray(raw["label"], dtype=np.float64).reshape(-1)
    if bvp.size == 0 or labels.size == 0:
        raise PpgDaliaLayoutError(f"{pkl_path.name}: empty BVP or label stream")
    if not np.all(np.isfinite(bvp)) or not np.all(np.isfinite(labels)):
        raise PpgDaliaLayoutError(f"{pkl_path.name}: non-finite values in BVP/label")
    if np.any(labels <= 0.0):
        raise PpgDaliaLayoutError(f"{pkl_path.name}: non-positive HR label (not plausible)")

    # Sampling-rate sanity: the two streams must describe the same recording duration.
    bvp_seconds = bvp.size / _BVP_FS_HZ
    label_seconds = labels.size / _LABEL_FS_HZ
    if abs(bvp_seconds - label_seconds) > _DURATION_TOLERANCE * max(bvp_seconds, label_seconds):
        raise PpgDaliaLayoutError(
            f"{pkl_path.name}: BVP duration {bvp_seconds:.0f}s vs label duration "
            f"{label_seconds:.0f}s disagree — sampling-rate constants may be wrong"
        )
    return bvp, labels


def _window_features(bvp: FloatArray, labels: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Cut BVP into the labels' own 8 s / 2 s-step windows; stat-summarise each."""
    win = int(_HR_WINDOW_SECONDS * _BVP_FS_HZ)
    step = int(_HR_STEP_SECONDS * _BVP_FS_HZ)
    feats: list[list[float]] = []
    targets: list[float] = []
    for i in range(labels.size):
        start = i * step
        end = start + win
        if end > bvp.size:
            break
        seg = bvp[start:end]
        feats.append([float(seg.mean()), float(seg.std()), float(seg.min()),
                      float(seg.max()), float(np.ptp(seg))])
        targets.append(float(labels[i]))
    if not feats:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.asarray(feats, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def load_ppg_dalia_hr_windows(
    root: Path,
    *,
    subjects: Sequence[str] | None = None,
    max_subjects: int | None = None,
    max_windows_per_subject: int | None = None,
) -> HrWindows:
    """Parse PPG-DaLiA subjects into a supervised (BVP-features → GT-HR) set.

    Aggregates across subjects (subject provenance retained). Raises
    `PpgDaliaLayoutError` on an invalid layout. `max_windows_per_subject` caps work
    for the fast smoke path.
    """
    files = _subject_files(root)
    if subjects is not None:
        wanted = set(subjects)
        files = [p for p in files if p.stem in wanted]
    if max_subjects is not None:
        files = files[:max_subjects]
    if not files:
        raise PpgDaliaLayoutError(f"no PPG-DaLiA subject pickles found under {root}")

    all_feats: list[FloatArray] = []
    all_targets: list[FloatArray] = []
    used: list[str] = []
    for pkl_path in files:
        bvp, labels = _load_subject(pkl_path)
        feats, targets = _window_features(bvp, labels)
        if max_windows_per_subject is not None:
            feats, targets = feats[:max_windows_per_subject], targets[:max_windows_per_subject]
        if len(targets) == 0:
            continue
        all_feats.append(feats)
        all_targets.append(targets)
        used.append(pkl_path.stem)

    if not all_feats:
        raise PpgDaliaLayoutError(f"no usable HR windows extracted from {root}")
    return HrWindows(
        features=np.concatenate(all_feats, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        feature_names=FEATURE_NAMES,
        subjects=tuple(used),
    )
