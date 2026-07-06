"""MLX training loop for the 1D-CNN PPG→HR encoder (docs/16 Sprint 10).

Trains the encoder+head on Apple-silicon Metal via MLX, then exports plain-NumPy
`EncoderWeights` so inference stays MLX-free (`encoder_model.py`). MLX is a guarded
import — this module loads without it and `train_encoder` raises
`TrainBackendUnavailable` if it is missing, mirroring `backends.MlxBackend`.

The loop reports every epoch through an `on_epoch` callback (train MSE + held-out
HR MAE/RMSE in bpm) so the caller can stream progress live — the trainer itself
prints nothing.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ai.training.backends import TrainBackendUnavailable
from ai.training.config import DEFAULT_SEED, set_global_seed
from ai.training.encoder_model import (
    CONV_CHANNELS,
    EMBEDDING_DIM,
    KERNEL_SIZE,
    PADDING,
    STRIDE,
    EncoderWeights,
    FloatArray,
    predict_hr,
    znorm_windows,
)


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder training hyperparameters (optimiser knobs, not clinical content).

    Defaults are tuned for a full-quality run (all data, hardware-maxing) — not a
    fast demo. `peak_lr` is the top of a warmup→cosine-decay schedule; the best
    validation checkpoint is kept, never the final epoch's weights.
    """

    seed: int = DEFAULT_SEED
    epochs: int = 200
    learning_rate: float = 2e-3  # peak LR (cosine-decayed after warmup)
    batch_size: int = 256
    val_fraction: float = 0.25  # fraction of SUBJECTS held out (see splits.py)
    dropout: float = 0.1
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05  # linear warmup share of total steps


@dataclass(frozen=True)
class EpochLog:
    epoch: int
    epochs_total: int
    train_mse: float  # standardised-target MSE (the optimised loss)
    val_mae_bpm: float
    val_rmse_bpm: float
    learning_rate: float
    is_best: bool
    seconds: float


ProgressCallback = Callable[[EpochLog], None]


@dataclass(frozen=True)
class TrainingHistory:
    logs: tuple[EpochLog, ...] = field(default_factory=tuple)
    best_epoch: int = 0

    @property
    def best_val_mae(self) -> float:
        return min((log.val_mae_bpm for log in self.logs), default=float("nan"))


def _standardise_targets(y: FloatArray) -> tuple[float, float]:
    mean = float(np.mean(y))
    std = float(np.std(y))
    return mean, (std if std > 1e-6 else 1.0)


def train_encoder(
    signals_train: FloatArray,
    y_train: FloatArray,
    signals_val: FloatArray,
    y_val: FloatArray,
    *,
    sample_rate_hz: float,
    window_samples: int,
    config: EncoderConfig | None = None,
    on_epoch: ProgressCallback | None = None,
) -> tuple[EncoderWeights, TrainingHistory]:
    """Train the conv encoder on Metal; return NumPy weights + per-epoch history."""
    cfg = config or EncoderConfig()
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
    except ImportError as exc:  # pragma: no cover - exercised only where MLX absent
        raise TrainBackendUnavailable(
            "training the biosignal encoder requires the `mlx` package (pip install mlx); "
            "it is Apple-silicon only. Inference does NOT need MLX."
        ) from exc

    set_global_seed(cfg.seed)
    hr_mean, hr_std = _standardise_targets(y_train)

    xtr = mx.array(znorm_windows(signals_train)[:, :, None].astype(np.float32))
    ytr = mx.array(((y_train - hr_mean) / hr_std).astype(np.float32))
    xva = mx.array(znorm_windows(signals_val)[:, :, None].astype(np.float32))

    class ConvHrEncoder(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            in_c = 1
            self.convs = []
            for out_c in CONV_CHANNELS:
                self.convs.append(
                    nn.Conv1d(in_c, out_c, KERNEL_SIZE, stride=STRIDE, padding=PADDING)
                )
                in_c = out_c
            self.dropout = nn.Dropout(cfg.dropout)
            self.head = nn.Linear(EMBEDDING_DIM, 1)

        def __call__(self, x: object) -> object:
            for conv in self.convs:
                x = nn.relu(conv(x))
            x = mx.mean(x, axis=1)  # global average pool -> [B, EMBEDDING_DIM]
            return self.head(self.dropout(x))[:, 0]

    model = ConvHrEncoder()
    mx.eval(model.parameters())

    n = int(xtr.shape[0])
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = max(1, int(total_steps * cfg.warmup_frac))
    schedule = optim.join_schedules(
        [
            optim.linear_schedule(0.0, cfg.learning_rate, warmup_steps),
            optim.cosine_decay(cfg.learning_rate, max(1, total_steps - warmup_steps)),
        ],
        [warmup_steps],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=cfg.weight_decay)

    def loss_fn(m: object, xb: object, yb: object) -> object:
        pred = m(xb)
        return mx.mean((pred - yb) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(cfg.seed)
    logs: list[EpochLog] = []
    best_mae = float("inf")
    best_epoch = 0

    def export() -> EncoderWeights:
        return _export_weights(
            model, mx, hr_mean=hr_mean, hr_std=hr_std,
            sample_rate_hz=sample_rate_hz, window_samples=window_samples,
        )

    best_weights = export()

    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        model.train()
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, cfg.batch_size):
            idx = mx.array(order[start : start + cfg.batch_size])
            xb = mx.take(xtr, idx, axis=0)
            yb = mx.take(ytr, idx, axis=0)
            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += float(loss)
            n_batches += 1

        model.eval()  # disable dropout for the held-out forward
        val_pred = np.array(model(xva)) * hr_std + hr_mean
        err = val_pred - y_val
        val_mae = float(np.mean(np.abs(err)))
        is_best = val_mae < best_mae
        if is_best:
            best_mae, best_epoch = val_mae, epoch
            best_weights = export()  # keep the BEST checkpoint, not the final epoch
        log = EpochLog(
            epoch=epoch,
            epochs_total=cfg.epochs,
            train_mse=epoch_loss / max(1, n_batches),
            val_mae_bpm=val_mae,
            val_rmse_bpm=float(np.sqrt(np.mean(err**2))),
            learning_rate=float(schedule(mx.array(epoch * steps_per_epoch))),
            is_best=is_best,
            seconds=time.time() - started,
        )
        logs.append(log)
        if on_epoch is not None:
            on_epoch(log)

    return best_weights, TrainingHistory(logs=tuple(logs), best_epoch=best_epoch)


def _export_weights(
    model: object,
    mx: object,
    *,
    hr_mean: float,
    hr_std: float,
    sample_rate_hz: float,
    window_samples: int,
) -> EncoderWeights:
    """Pull MLX parameters into a backend-agnostic NumPy `EncoderWeights`."""
    conv_w: list[FloatArray] = []
    conv_b: list[FloatArray] = []
    for conv in model.convs:  # type: ignore[attr-defined]
        conv_w.append(np.array(conv.weight, dtype=np.float64))  # [C_out, K, C_in]
        conv_b.append(np.array(conv.bias, dtype=np.float64))  # [C_out]
    head_w = np.array(model.head.weight, dtype=np.float64)[0]  # type: ignore[attr-defined]
    head_b = float(np.array(model.head.bias, dtype=np.float64)[0])  # type: ignore[attr-defined]
    weights = EncoderWeights(
        conv_w=tuple(conv_w),
        conv_b=tuple(conv_b),
        head_w=head_w,
        head_b=head_b,
        hr_mean=hr_mean,
        hr_std=hr_std,
        sample_rate_hz=sample_rate_hz,
        window_samples=window_samples,
    )
    return weights


def predict_hr_numpy(weights: EncoderWeights, signals: FloatArray) -> FloatArray:
    """Convenience re-export: NumPy inference (see `encoder_model.predict_hr`)."""
    return predict_hr(weights, signals)
