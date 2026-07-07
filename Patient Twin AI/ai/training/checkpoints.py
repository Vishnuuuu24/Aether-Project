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
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai.training.backends import TrainedHead
from ai.training.config import TrainConfig
from ai.training.encoder_model import EncoderWeights
from core.versioning import VersionRegistry

DEFAULT_CHECKPOINT_ROOT = Path("checkpoints")


@dataclass(frozen=True)
class CheckpointHandle:
    path: Path
    version: str
    manifest: dict[str, object]


def _hash_identity(identity: dict[str, object]) -> str:
    blob = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _version_id(name: str, config: TrainConfig, provenance: dict[str, object]) -> str:
    """Deterministic content-addressed id: hash of run identity (no timestamp)."""
    identity = {"name": name, "config": asdict(config), "provenance": provenance}
    return f"{name}@{_hash_identity(identity)}"


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


def write_encoder_checkpoint(
    weights: EncoderWeights,
    *,
    name: str,
    config: object,
    provenance: dict[str, object],
    metrics: dict[str, float] | None = None,
    root: Path = DEFAULT_CHECKPOINT_ROOT,
    now: datetime | None = None,
) -> CheckpointHandle:
    """Write a conv-encoder checkpoint (`encoder.npz` + `manifest.json`).

    Same content-addressed identity scheme as `write_checkpoint`, but serialises the
    multi-array `EncoderWeights` (variable conv depth) rather than a single linear
    head. `config` is any dataclass of hyperparameters (hashed into the version).
    """
    identity = {"name": name, "config": asdict(config), "provenance": provenance}  # type: ignore[call-overload]
    version = f"{name}@{_hash_identity(identity)}"
    out_dir = root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {"n_conv": np.array([len(weights.conv_w)])}
    for i, (w, b) in enumerate(zip(weights.conv_w, weights.conv_b, strict=True)):
        arrays[f"conv{i}_w"] = w
        arrays[f"conv{i}_b"] = b
    arrays["head_w"] = weights.head_w
    arrays["head_b"] = np.array([weights.head_b])
    arrays["hr_mean"] = np.array([weights.hr_mean])
    arrays["hr_std"] = np.array([weights.hr_std])
    arrays["sample_rate_hz"] = np.array([weights.sample_rate_hz])
    arrays["window_samples"] = np.array([weights.window_samples])
    np.savez(out_dir / "encoder.npz", **arrays)

    manifest: dict[str, object] = {
        "name": name,
        "version": version,
        "kind": "conv_hr_encoder",
        "created_at": (now or datetime.now(UTC)).isoformat(),
        "config": asdict(config),  # type: ignore[call-overload]
        "provenance": provenance,
        "metrics": metrics or {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return CheckpointHandle(path=out_dir, version=version, manifest=manifest)


def load_encoder_weights(handle_or_path: CheckpointHandle | Path) -> EncoderWeights:
    """Reload `EncoderWeights` from a checkpoint dir — NumPy only, no MLX needed."""
    path = handle_or_path.path if isinstance(handle_or_path, CheckpointHandle) else handle_or_path
    npz = np.load(path / "encoder.npz", allow_pickle=False)
    n_conv = int(npz["n_conv"][0])
    conv_w = tuple(npz[f"conv{i}_w"].astype(np.float64) for i in range(n_conv))
    conv_b = tuple(npz[f"conv{i}_b"].astype(np.float64) for i in range(n_conv))
    return EncoderWeights(
        conv_w=conv_w,
        conv_b=conv_b,
        head_w=npz["head_w"].astype(np.float64),
        head_b=float(npz["head_b"][0]),
        hr_mean=float(npz["hr_mean"][0]),
        hr_std=float(npz["hr_std"][0]),
        sample_rate_hz=float(npz["sample_rate_hz"][0]),
        window_samples=int(npz["window_samples"][0]),
    )


def write_papagei_checkpoint(
    weights: object,
    *,
    name: str,
    config: object,
    provenance: dict[str, object],
    metrics: dict[str, float] | None = None,
    root: Path = DEFAULT_CHECKPOINT_ROOT,
    now: datetime | None = None,
) -> CheckpointHandle:
    """Write a fine-tuned PaPaGei-S trunk + HR head (`papagei.npz` + `manifest.json`).

    Serialises only the *learned* tensors of `PapageiEncoderWeights`; the per-block
    geometry (channels/stride/downsample) is deterministic and recomputed on load via
    `block_geometry()`. Same content-addressed identity scheme as the other writers.
    """
    from ai.training.papagei_resnet import BatchNormParams, PapageiEncoderWeights

    assert isinstance(weights, PapageiEncoderWeights)
    is_instance = is_dataclass(config) and not isinstance(config, type)
    config_dict = asdict(config) if is_instance else config
    identity = {"name": name, "config": config_dict, "provenance": provenance}
    version = f"{name}@{_hash_identity(identity)}"
    out_dir = root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    def _bn_arrays(prefix: str, bn: BatchNormParams) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_gamma": bn.gamma, f"{prefix}_beta": bn.beta,
            f"{prefix}_mean": bn.running_mean, f"{prefix}_var": bn.running_var,
        }

    arrays: dict[str, np.ndarray] = {
        "first_conv_w": weights.first_conv_w,
        "first_conv_b": weights.first_conv_b,
        "head_w": weights.head_w,
        "n_blocks": np.array([len(weights.blocks)]),
    }
    arrays.update(_bn_arrays("first_bn", weights.first_bn))
    arrays.update(_bn_arrays("final_bn", weights.final_bn))
    for i, blk in enumerate(weights.blocks):
        arrays.update(_bn_arrays(f"b{i}_bn1", blk.bn1))
        arrays.update(_bn_arrays(f"b{i}_bn2", blk.bn2))
        arrays[f"b{i}_conv1_w"] = blk.conv1_w
        arrays[f"b{i}_conv1_b"] = blk.conv1_b
        arrays[f"b{i}_conv2_w"] = blk.conv2_w
        arrays[f"b{i}_conv2_b"] = blk.conv2_b
    np.savez(out_dir / "papagei.npz", **arrays)

    manifest: dict[str, object] = {
        "name": name,
        "version": version,
        "kind": "papagei_s_hr_encoder",
        "created_at": (now or datetime.now(UTC)).isoformat(),
        "config": config_dict,
        "provenance": provenance,
        "metrics": metrics or {},
        "head_b": weights.head_b,
        "hr_mean": weights.hr_mean,
        "hr_std": weights.hr_std,
        "sample_rate_hz": weights.sample_rate_hz,
        "window_samples": weights.window_samples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return CheckpointHandle(path=out_dir, version=version, manifest=manifest)


def load_papagei_weights(handle_or_path: CheckpointHandle | Path) -> object:
    """Reload a fine-tuned `PapageiEncoderWeights` — NumPy only, no torch/MLX needed."""
    from ai.training.papagei_resnet import (
        BatchNormParams,
        BlockParams,
        PapageiEncoderWeights,
        block_geometry,
    )

    path = handle_or_path.path if isinstance(handle_or_path, CheckpointHandle) else handle_or_path
    manifest = json.loads((path / "manifest.json").read_text())
    npz = np.load(path / "papagei.npz", allow_pickle=False)

    def _bn(prefix: str) -> BatchNormParams:
        return BatchNormParams(
            gamma=npz[f"{prefix}_gamma"].astype(np.float64),
            beta=npz[f"{prefix}_beta"].astype(np.float64),
            running_mean=npz[f"{prefix}_mean"].astype(np.float64),
            running_var=npz[f"{prefix}_var"].astype(np.float64),
        )

    schedule = block_geometry()
    blocks: list[BlockParams] = []
    for i, meta in enumerate(schedule):
        blocks.append(
            BlockParams(
                is_first_block=meta.is_first_block, downsample=meta.downsample,
                stride=meta.stride, in_channels=meta.in_channels,
                out_channels=meta.out_channels,
                bn1=_bn(f"b{i}_bn1"), conv1_w=npz[f"b{i}_conv1_w"].astype(np.float64),
                conv1_b=npz[f"b{i}_conv1_b"].astype(np.float64),
                bn2=_bn(f"b{i}_bn2"), conv2_w=npz[f"b{i}_conv2_w"].astype(np.float64),
                conv2_b=npz[f"b{i}_conv2_b"].astype(np.float64),
            )
        )
    return PapageiEncoderWeights(
        first_conv_w=npz["first_conv_w"].astype(np.float64),
        first_conv_b=npz["first_conv_b"].astype(np.float64),
        first_bn=_bn("first_bn"),
        blocks=tuple(blocks),
        final_bn=_bn("final_bn"),
        head_w=npz["head_w"].astype(np.float64),
        head_b=float(manifest["head_b"]),
        hr_mean=float(manifest["hr_mean"]),
        hr_std=float(manifest["hr_std"]),
        sample_rate_hz=float(manifest["sample_rate_hz"]),
        window_samples=int(manifest["window_samples"]),
    )


def write_stress_head_checkpoint(
    head: object,
    *,
    name: str,
    config: object,
    provenance: dict[str, object],
    metrics: dict[str, float] | None = None,
    root: Path = DEFAULT_CHECKPOINT_ROOT,
    now: datetime | None = None,
) -> CheckpointHandle:
    """Write a stress-head checkpoint (`stress_head.npz` + `manifest.json`).

    The stress head is a logistic regression on the encoder embedding — a small NumPy
    artifact, content-addressed on the same identity scheme as the other checkpoints.
    """
    from ai.training.stress_head import StressHead

    assert isinstance(head, StressHead)
    is_instance = is_dataclass(config) and not isinstance(config, type)
    config_dict = asdict(config) if is_instance else config
    identity = {"name": name, "config": config_dict, "provenance": provenance}
    version = f"{name}@{_hash_identity(identity)}"
    out_dir = root / version
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / "stress_head.npz",
        w=head.w,
        b=np.array([head.b]),
        feat_mean=head.feat_mean,
        feat_std=head.feat_std,
    )
    manifest: dict[str, object] = {
        "name": name,
        "version": version,
        "kind": "ppg_stress_head",
        "head_version": head.version,
        "created_at": (now or datetime.now(UTC)).isoformat(),
        "config": config_dict,
        "provenance": provenance,
        "metrics": metrics or {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return CheckpointHandle(path=out_dir, version=version, manifest=manifest)


def load_stress_head(handle_or_path: CheckpointHandle | Path) -> object:
    """Reload a `StressHead` from a checkpoint dir — NumPy only, no MLX needed."""
    from ai.training.stress_head import StressHead

    path = handle_or_path.path if isinstance(handle_or_path, CheckpointHandle) else handle_or_path
    manifest = json.loads((path / "manifest.json").read_text())
    npz = np.load(path / "stress_head.npz", allow_pickle=False)
    return StressHead(
        w=npz["w"].astype(np.float64),
        b=float(npz["b"][0]),
        feat_mean=npz["feat_mean"].astype(np.float64),
        feat_std=npz["feat_std"].astype(np.float64),
        version=str(manifest.get("head_version", "ppg-stress-logreg-v1")),
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
