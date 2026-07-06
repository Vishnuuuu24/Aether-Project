"""Versioned checkpoint writer (docs/16 Sprint 9; docs/04 §7 versioning/audit).

Every trained artifact is written with a **content-addressed version** and a
manifest recording provenance (dataset, subjects, hyperparameters, seed, metrics).
The version id is a hash of the run's identity — deterministic, so the same run
reproduces the same id (the wall-clock `created_at` is recorded but excluded from
the hash).

`register_checkpoint_version` stamps that version onto a `VersionRegistry` by
returning a NEW registry via `with_versions(...)` — it does NOT mutate any global.
Promoting a learned model is a human-gated versioned release (CLAUDE.md principle
5); this writer produces the artifact + the stamped registry, and the promotion
decision stays with a human running the eval gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai.training.backends import TrainedHead
from ai.training.config import TrainConfig
from core.versioning import VersionRegistry

DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")


@dataclass(frozen=True)
class CheckpointHandle:
    path: Path
    version: str
    manifest: dict[str, object]


def _version_id(name: str, config: TrainConfig, provenance: dict[str, object]) -> str:
    """Deterministic content-addressed id: hash of run identity (no timestamp)."""
    identity = {
        "name": name,
        "config": asdict(config),
        "provenance": provenance,
    }
    blob = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return f"{name}@{hashlib.sha256(blob).hexdigest()[:12]}"


def write_checkpoint(
    head: TrainedHead,
    *,
    name: str,
    config: TrainConfig,
    provenance: dict[str, object],
    metrics: dict[str, float] | None = None,
    root: Path = DEFAULT_CHECKPOINT_ROOT,
    now: datetime | None = None,
) -> CheckpointHandle:
    """Write `<root>/<version>/` with `head.npz` + `manifest.json`; return a handle.

    `provenance` is the reproducibility record (dataset, subjects, feature names,
    sample count). `metrics` (if scored) is stored for audit but excluded from the
    version hash so scoring never changes an artifact's identity.
    """
    version = _version_id(name, config, provenance)
    out_dir = root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "head.npz",
        weights=head.weights,
        bias=np.array([head.bias]),
        feature_mean=head.feature_mean,
        feature_std=head.feature_std,
        feature_names=np.array(head.feature_names),
    )

    manifest: dict[str, object] = {
        "name": name,
        "version": version,
        "backend": head.backend,
        "created_at": (now or datetime.now(UTC)).isoformat(),
        "config": asdict(config),
        "provenance": provenance,
        "metrics": metrics or {},
        "feature_names": list(head.feature_names),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return CheckpointHandle(path=out_dir, version=version, manifest=manifest)


def load_head(handle_or_path: CheckpointHandle | Path) -> TrainedHead:
    """Reload a `TrainedHead` from a checkpoint dir — no MLX/CUDA needed (NumPy only)."""
    path = handle_or_path.path if isinstance(handle_or_path, CheckpointHandle) else handle_or_path
    manifest = json.loads((path / "manifest.json").read_text())
    npz = np.load(path / "head.npz", allow_pickle=False)
    return TrainedHead(
        weights=npz["weights"].astype(np.float64),
        bias=float(npz["bias"][0]),
        feature_mean=npz["feature_mean"].astype(np.float64),
        feature_std=npz["feature_std"].astype(np.float64),
        feature_names=tuple(str(n) for n in npz["feature_names"].tolist()),
        backend=str(manifest["backend"]),
    )


def register_checkpoint_version(
    version: str,
    *,
    field: str = "baseline_engine",
    base: VersionRegistry | None = None,
) -> VersionRegistry:
    """Stamp `version` onto a NEW registry (human-gated release model; no mutation).

    A learned biosignal encoder implements `BaselineEngine`/`FeatureExtractor`, so it
    stamps the `baseline_engine` version field by default (docs/04 §6)."""
    registry = base or VersionRegistry.from_env()
    return registry.with_versions(**{field: version})
