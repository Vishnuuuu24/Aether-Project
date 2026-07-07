"""WESAD wrist-BVP deviation eval CLI — runs the classical arm on a synthetic subject
and fails safe when the dataset is absent (docs/16 Sprint 10)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ai.eval_datasets.wesad import WESAD_BASELINE_LABEL, WESAD_STRESS_LABEL
from ai.training.wesad_deviation_eval import run

_FS = 700.0
_WRIST_FS = 64.0


def _bvp_at(bpm: float, seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * _WRIST_FS)
    t = np.arange(n) / _WRIST_FS
    sig = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm
    while beat < seconds:
        sig += np.exp(-((t - beat) ** 2) / (2 * 0.040**2))
        beat += (60.0 / bpm) * float(rng.uniform(0.92, 1.08))
    return sig


def _write_subject(root: Path) -> None:
    base_secs, stress_secs = 240.0, 160.0
    bvp = np.concatenate([_bvp_at(65.0, base_secs, 1), _bvp_at(100.0, stress_secs, 2)])
    labels = np.concatenate(
        [
            np.full(int(base_secs * _FS), WESAD_BASELINE_LABEL, dtype=int),
            np.full(int(stress_secs * _FS), WESAD_STRESS_LABEL, dtype=int),
        ]
    )
    sub = root / "S99"
    sub.mkdir(parents=True)
    with (sub / "S99.pkl").open("wb") as fh:
        pickle.dump({"signal": {"wrist": {"BVP": bvp.reshape(-1, 1)}}, "label": labels}, fh)


def test_run_classical_arm_on_synthetic_subject(tmp_path: Path) -> None:
    _write_subject(tmp_path)
    assert run(tmp_path, checkpoint=None, window_seconds=8.0) == 0


def test_run_fails_safe_when_absent(tmp_path: Path) -> None:
    assert run(tmp_path, checkpoint=None) == 2
