"""On-device training harness (docs/16).

One codebase, config-switched: `TRAIN_BACKEND` picks `mlx` (Mac) or `cuda_qlora`
(server). Data prep, checkpointing, and eval are shared across both backends; only
the inner training loop differs. A learned model is a new implementation of a stable
interface (`FeatureExtractor` / `BaselineEngine` / `Retriever`), gated on the
existing eval harness before it can replace the classical path — never a new call
site (CLAUDE.md).
"""

from __future__ import annotations

from ai.training.backends import (
    CudaQloraBackend,
    MlxBackend,
    TrainBackend,
    TrainBackendUnavailable,
    TrainedHead,
    select_backend,
)
from ai.training.checkpoints import (
    CheckpointHandle,
    load_encoder_weights,
    load_head,
    register_checkpoint_version,
    write_checkpoint,
    write_encoder_checkpoint,
)
from ai.training.config import TrainConfig, set_global_seed
from ai.training.encoder_model import EncoderWeights, predict_hr
from ai.training.splits import subject_held_out_split

__all__ = [
    "CheckpointHandle",
    "CudaQloraBackend",
    "EncoderWeights",
    "MlxBackend",
    "TrainBackend",
    "TrainBackendUnavailable",
    "TrainConfig",
    "TrainedHead",
    "load_encoder_weights",
    "load_head",
    "predict_hr",
    "register_checkpoint_version",
    "select_backend",
    "set_global_seed",
    "subject_held_out_split",
    "write_checkpoint",
    "write_encoder_checkpoint",
]
