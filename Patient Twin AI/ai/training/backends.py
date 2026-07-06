"""Training backend seam (docs/16 Sprint 9; CLAUDE.md training table).

ONE codebase, config-switched: the inner training loop is the only thing that
differs between the Mac and the server. `TRAIN_BACKEND` selects it —

    mlx         → Apple-silicon Metal training (this Mac)          MlxBackend
    cuda_qlora  → NVIDIA CUDA training (the H200 slice)            CudaQloraBackend

Data prep, checkpointing, and evaluation are SHARED across both (they live in the
sibling modules); backends only turn `(features, targets)` into a `TrainedHead`.

A `TrainedHead` is a plain NumPy linear model, so inference / the classical
fallback never needs MLX or CUDA — only *training* does. MLX is a guarded import:
this module loads without it, and `MlxBackend.fit` raises a clear
`TrainBackendUnavailable` if it is missing (docs/16: skip-guard when absent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ai.training.config import TrainConfig, set_global_seed

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

MLX_BACKEND = "mlx"
CUDA_QLORA_BACKEND = "cuda_qlora"


class TrainBackendUnavailable(RuntimeError):
    """Raised when the selected backend cannot run on this host (missing MLX / CUDA)."""


@dataclass(frozen=True)
class TrainedHead:
    """A standardised linear head: `y ≈ ((x - mean) / std) · weights + bias`.

    Standardisation stats are baked in so the head is self-contained and
    backend-agnostic on load."""

    weights: FloatArray  # [F]
    bias: float
    feature_mean: FloatArray  # [F]
    feature_std: FloatArray  # [F]
    feature_names: tuple[str, ...]
    backend: str

    def predict(self, features: FloatArray) -> FloatArray:
        z = (features - self.feature_mean) / self.feature_std
        return z @ self.weights + self.bias


@runtime_checkable
class TrainBackend(Protocol):
    name: str

    def fit(
        self, features: FloatArray, targets: FloatArray, config: TrainConfig
    ) -> TrainedHead: ...


def _standardise(features: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0.0] = 1.0  # guard constant columns
    return mean, std


class MlxBackend:
    """Linear-head training via full-batch gradient descent on MLX (Metal)."""

    name = MLX_BACKEND

    def fit(self, features: FloatArray, targets: FloatArray, config: TrainConfig) -> TrainedHead:
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - exercised only where MLX absent
            raise TrainBackendUnavailable(
                "backend 'mlx' requires the `mlx` package (pip install mlx); "
                "it is Apple-silicon only. Set TRAIN_BACKEND=cuda_qlora on the server."
            ) from exc

        set_global_seed(config.seed)
        mean, std = _standardise(features)
        x = mx.array(((features - mean) / std).astype(np.float32))
        y = mx.array(targets.astype(np.float32))
        n_features = x.shape[1]
        w = mx.zeros((n_features,))
        b = mx.zeros((1,))

        def loss_fn(w: Any, b: Any) -> Any:
            pred = x @ w + b
            return mx.mean((pred - y) ** 2)

        grad_fn = mx.grad(loss_fn, argnums=(0, 1))
        lr = config.learning_rate
        for _ in range(config.epochs):
            gw, gb = grad_fn(w, b)
            w = w - lr * gw
            b = b - lr * gb
            mx.eval(w, b)

        return TrainedHead(
            weights=np.array(w, dtype=np.float64),
            bias=float(np.array(b, dtype=np.float64)[0]),
            feature_mean=mean,
            feature_std=std,
            feature_names=(),  # filled by the caller that knows the dataset
            backend=self.name,
        )


class CudaQloraBackend:
    """Server (H200) backend. A REAL class that refuses to run without CUDA rather
    than silently degrading — QLoRA needs bitsandbytes/PEFT, which are CUDA-only
    (CLAUDE.md). Wired now so the seam is complete; the training loop lands with the
    server track."""

    name = CUDA_QLORA_BACKEND

    def fit(self, features: FloatArray, targets: FloatArray, config: TrainConfig) -> TrainedHead:
        if not _cuda_available():
            raise TrainBackendUnavailable(
                "backend 'cuda_qlora' requires a CUDA GPU (the H200 slice); "
                "not available on this host. Use TRAIN_BACKEND=mlx on the Mac."
            )
        raise NotImplementedError(  # pragma: no cover - deferred to the server track
            "cuda_qlora training loop is deferred to the H200 server track (docs/16)."
        )


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def select_backend(name: str | None = None) -> TrainBackend:
    """Resolve a backend by name (defaults to `TrainConfig().backend`, i.e. the
    `TRAIN_BACKEND` env var). Unknown names fail loudly."""
    resolved = name or TrainConfig().backend
    if resolved == MLX_BACKEND:
        return MlxBackend()
    if resolved == CUDA_QLORA_BACKEND:
        return CudaQloraBackend()
    raise ValueError(
        f"unknown TRAIN_BACKEND {resolved!r}; supported: {MLX_BACKEND}, {CUDA_QLORA_BACKEND}"
    )
