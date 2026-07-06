"""Deterministic training config + seeding (docs/16 Sprint 9).

Every run is reproducible: one frozen `TrainConfig` carries the seed and
hyperparameters, and `set_global_seed` pins Python / NumPy / MLX RNGs together.
Nothing here is clinical content — these are optimiser knobs.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np

DEFAULT_SEED = 20260101


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters + provenance knobs for one training run.

    `backend` selects the compute path (`mlx` on the Mac, `cuda_qlora` on the
    server); it defaults to the `TRAIN_BACKEND` env var so where-it-runs is
    configuration, not code (CLAUDE.md, one codebase config-switched).
    """

    seed: int = DEFAULT_SEED
    epochs: int = 200
    learning_rate: float = 0.05
    val_fraction: float = 0.2
    backend: str = field(default_factory=lambda: os.environ.get("TRAIN_BACKEND", "mlx"))

    def __post_init__(self) -> None:
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")


def set_global_seed(seed: int) -> None:
    """Seed every RNG the training loop can touch. MLX is seeded only if present so
    this module imports with no hard MLX dependency (docs/16: guarded import)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import mlx.core as mx
    except ImportError:
        return
    mx.random.seed(seed)
