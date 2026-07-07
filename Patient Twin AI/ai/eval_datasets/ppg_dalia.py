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
_ACC_FS_HZ = 32.0  # Empatica E4 wrist accelerometer (3-axis)
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


@dataclass(frozen=True)
class SignalWindows:
    """Raw supervised HR set for the biosignal encoder (docs/16 Sprint 10).

    Unlike `HrWindows` (5 reduced stats), this keeps the RAW BVP window `signals`
    [N, L] so a convolutional encoder can learn HR from the waveform morphology.
    `subject_ids` is per-window provenance (length N) — the encoder trainer splits
    on it to guarantee no subject leaks across train/val (docs/16 Sprint 10 Don't).
    """

    signals: FloatArray  # [N, L] raw BVP amplitude windows
    targets: FloatArray  # [N] ground-truth HR (bpm)
    subject_ids: tuple[str, ...]  # per-window subject stem, length N
    sample_rate_hz: float
    window_samples: int

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    @property
    def subjects(self) -> tuple[str, ...]:
        """Distinct contributing subjects, in first-seen order (provenance)."""
        seen: dict[str, None] = {}
        for sid in self.subject_ids:
            seen.setdefault(sid, None)
        return tuple(seen)


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


def _window_signals(bvp: FloatArray, labels: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Cut BVP into the labels' own 8 s / 2 s-step windows; keep the RAW segment."""
    win = int(_HR_WINDOW_SECONDS * _BVP_FS_HZ)
    step = int(_HR_STEP_SECONDS * _BVP_FS_HZ)
    segs: list[FloatArray] = []
    targets: list[float] = []
    for i in range(labels.size):
        start = i * step
        end = start + win
        if end > bvp.size:
            break
        segs.append(bvp[start:end])
        targets.append(float(labels[i]))
    if not segs:
        return np.empty((0, win), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.asarray(segs, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _load_subject_fused(pkl_path: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Load (wrist_bvp @64 Hz, wrist_acc [Nacc,3] @32 Hz, gt_hr_labels) with validation."""
    bvp, labels = _load_subject(pkl_path)
    with pkl_path.open("rb") as fh:
        raw: Any = pickle.load(fh, encoding="latin1")  # noqa: S301 - trusted local dataset
    wrist = raw["signal"]["wrist"]
    if "ACC" not in wrist:
        raise PpgDaliaLayoutError(f"{pkl_path.name}: wrist block has no 'ACC'")
    acc = np.asarray(wrist["ACC"], dtype=np.float64)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise PpgDaliaLayoutError(f"{pkl_path.name}: wrist ACC not [N,3] (got {acc.shape})")
    if not np.all(np.isfinite(acc)):
        raise PpgDaliaLayoutError(f"{pkl_path.name}: non-finite wrist ACC")
    # ACC (~32 Hz) must describe the same duration as BVP (~64 Hz) within tolerance.
    acc_seconds = acc.shape[0] / _ACC_FS_HZ
    bvp_seconds = bvp.size / _BVP_FS_HZ
    if abs(acc_seconds - bvp_seconds) > _DURATION_TOLERANCE * max(acc_seconds, bvp_seconds):
        raise PpgDaliaLayoutError(
            f"{pkl_path.name}: ACC duration {acc_seconds:.0f}s vs BVP {bvp_seconds:.0f}s disagree"
        )
    return bvp, acc, labels


def _window_fused(
    bvp: FloatArray, acc: FloatArray, labels: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Cut BVP + ACC into the labels' 8 s / 2 s windows; ACC linearly resampled to the
    BVP grid so each fused window is [L, 4] = (BVP, acc_x, acc_y, acc_z)."""
    win = int(_HR_WINDOW_SECONDS * _BVP_FS_HZ)
    step = int(_HR_STEP_SECONDS * _BVP_FS_HZ)
    win_acc = int(_HR_WINDOW_SECONDS * _ACC_FS_HZ)
    step_acc = int(_HR_STEP_SECONDS * _ACC_FS_HZ)
    grid_bvp = np.linspace(0.0, 1.0, win)
    grid_acc = np.linspace(0.0, 1.0, win_acc)
    segs: list[FloatArray] = []
    targets: list[float] = []
    for i in range(labels.size):
        start, end = i * step, i * step + win
        a_start, a_end = i * step_acc, i * step_acc + win_acc
        if end > bvp.size or a_end > acc.shape[0]:
            break
        acc_seg = acc[a_start:a_end]  # [win_acc, 3]
        acc_up = np.stack(
            [np.interp(grid_bvp, grid_acc, acc_seg[:, c]) for c in range(3)], axis=1
        )  # [win, 3]
        fused = np.concatenate([bvp[start:end][:, None], acc_up], axis=1)  # [win, 4]
        segs.append(fused)
        targets.append(float(labels[i]))
    if not segs:
        return np.empty((0, win, 4), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.asarray(segs, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def load_ppg_dalia_hr_fused_windows(
    root: Path,
    *,
    subjects: Sequence[str] | None = None,
    max_subjects: int | None = None,
    max_windows_per_subject: int | None = None,
) -> SignalWindows:
    """Like `load_ppg_dalia_hr_signal_windows` but each window is `[L, 4]` — raw BVP
    plus the 3-axis wrist accelerometer resampled to the BVP grid (docs/16 Sprint 10
    fusion experiment). `signals` is therefore `[N, L, 4]`; everything else matches."""
    files = _subject_files(root)
    if subjects is not None:
        wanted = set(subjects)
        files = [p for p in files if p.stem in wanted]
    if max_subjects is not None:
        files = files[:max_subjects]
    if not files:
        raise PpgDaliaLayoutError(f"no PPG-DaLiA subject pickles found under {root}")

    win = int(_HR_WINDOW_SECONDS * _BVP_FS_HZ)
    all_signals: list[FloatArray] = []
    all_targets: list[FloatArray] = []
    subject_ids: list[str] = []
    for pkl_path in files:
        bvp, acc, labels = _load_subject_fused(pkl_path)
        segs, targets = _window_fused(bvp, acc, labels)
        if max_windows_per_subject is not None:
            segs, targets = segs[:max_windows_per_subject], targets[:max_windows_per_subject]
        if len(targets) == 0:
            continue
        all_signals.append(segs)
        all_targets.append(targets)
        subject_ids.extend([pkl_path.stem] * len(targets))

    if not all_signals:
        raise PpgDaliaLayoutError(f"no usable fused HR windows extracted from {root}")
    return SignalWindows(
        signals=np.concatenate(all_signals, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        subject_ids=tuple(subject_ids),
        sample_rate_hz=_BVP_FS_HZ,
        window_samples=win,
    )


def _window_papagei(
    bvp_125: FloatArray, labels: FloatArray, *, target_fs: float, segment_seconds: float
) -> tuple[FloatArray, FloatArray]:
    """Cut resampled (125 Hz) BVP into `segment_seconds` windows CENTRED on each GT-HR
    label's own 8 s window, so PaPaGei sees its native-length segment centred on the
    region the label describes. `bvp_125` is already resampled to `target_fs`."""
    seg = int(segment_seconds * target_fs)
    segs: list[FloatArray] = []
    targets: list[float] = []
    for i in range(labels.size):
        # label i summarises original [i*2s, i*2s+8s]; centre a seg-long window on it.
        centre_s = i * _HR_STEP_SECONDS + _HR_WINDOW_SECONDS / 2.0
        start = int(round((centre_s - segment_seconds / 2.0) * target_fs))
        end = start + seg
        if start < 0 or end > bvp_125.size:
            continue
        segs.append(bvp_125[start:end])
        targets.append(float(labels[i]))
    if not segs:
        return np.empty((0, seg), dtype=np.float64), np.empty(0, dtype=np.float64)
    return np.asarray(segs, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def load_ppg_dalia_hr_papagei_windows(
    root: Path,
    *,
    subjects: Sequence[str] | None = None,
    max_subjects: int | None = None,
    max_windows_per_subject: int | None = None,
    target_fs_hz: float = 125.0,
    segment_seconds: float = 10.0,
) -> SignalWindows:
    """PPG-DaLiA raw-BVP windows resampled to PaPaGei-S's pretrained input contract
    (default 125 Hz / 10 s = 1250 samples), each carrying the GT HR of the 8 s label it
    is centred on (docs/16 Sprint 10 pretrained-encoder init). Subject provenance kept
    for held-out splitting. Separate from `load_ppg_dalia_hr_signal_windows` (64 Hz /
    8 s, the from-scratch encoder's geometry) so both paths stay reproducible."""
    from ai.eval_datasets._resample import resample_poly_to

    files = _subject_files(root)
    if subjects is not None:
        wanted = set(subjects)
        files = [p for p in files if p.stem in wanted]
    if max_subjects is not None:
        files = files[:max_subjects]
    if not files:
        raise PpgDaliaLayoutError(f"no PPG-DaLiA subject pickles found under {root}")

    seg = int(segment_seconds * target_fs_hz)
    all_signals: list[FloatArray] = []
    all_targets: list[FloatArray] = []
    subject_ids: list[str] = []
    for pkl_path in files:
        bvp, labels = _load_subject(pkl_path)
        bvp_125 = resample_poly_to(bvp, _BVP_FS_HZ, target_fs_hz)
        segs, targets = _window_papagei(
            bvp_125, labels, target_fs=target_fs_hz, segment_seconds=segment_seconds
        )
        if max_windows_per_subject is not None:
            segs, targets = segs[:max_windows_per_subject], targets[:max_windows_per_subject]
        if len(targets) == 0:
            continue
        all_signals.append(segs)
        all_targets.append(targets)
        subject_ids.extend([pkl_path.stem] * len(targets))

    if not all_signals:
        raise PpgDaliaLayoutError(f"no usable PaPaGei HR windows extracted from {root}")
    return SignalWindows(
        signals=np.concatenate(all_signals, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        subject_ids=tuple(subject_ids),
        sample_rate_hz=target_fs_hz,
        window_samples=seg,
    )


def load_ppg_dalia_hr_signal_windows(
    root: Path,
    *,
    subjects: Sequence[str] | None = None,
    max_subjects: int | None = None,
    max_windows_per_subject: int | None = None,
) -> SignalWindows:
    """Parse PPG-DaLiA subjects into a RAW-BVP-window → GT-HR set (Sprint 10 encoder).

    Same window geometry and validation as `load_ppg_dalia_hr_windows`, but retains
    the raw 8 s BVP segment (not reduced stats) and tags each window with its subject
    stem so the trainer can hold whole subjects out. Raises `PpgDaliaLayoutError` on
    an invalid layout.
    """
    files = _subject_files(root)
    if subjects is not None:
        wanted = set(subjects)
        files = [p for p in files if p.stem in wanted]
    if max_subjects is not None:
        files = files[:max_subjects]
    if not files:
        raise PpgDaliaLayoutError(f"no PPG-DaLiA subject pickles found under {root}")

    win = int(_HR_WINDOW_SECONDS * _BVP_FS_HZ)
    all_signals: list[FloatArray] = []
    all_targets: list[FloatArray] = []
    subject_ids: list[str] = []
    for pkl_path in files:
        bvp, labels = _load_subject(pkl_path)
        segs, targets = _window_signals(bvp, labels)
        if max_windows_per_subject is not None:
            segs, targets = segs[:max_windows_per_subject], targets[:max_windows_per_subject]
        if len(targets) == 0:
            continue
        all_signals.append(segs)
        all_targets.append(targets)
        subject_ids.extend([pkl_path.stem] * len(targets))

    if not all_signals:
        raise PpgDaliaLayoutError(f"no usable HR signal windows extracted from {root}")
    return SignalWindows(
        signals=np.concatenate(all_signals, axis=0),
        targets=np.concatenate(all_targets, axis=0),
        subject_ids=tuple(subject_ids),
        sample_rate_hz=_BVP_FS_HZ,
        window_samples=win,
    )
