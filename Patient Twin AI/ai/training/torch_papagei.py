"""PyTorch/MPS fine-tuning of the PaPaGei-S trunk (docs/16 Sprint 10, pretrained init).

CLAUDE.md sanctions PyTorch-MPS as a Mac training backend for exactly this — the
~6M-param PaPaGei-S biosignal encoder. We FULL-fine-tune the pretrained trunk (user's
explicit choice, matching the DoD's "fine-tune the pretrained encoder") plus a fresh HR
head, then export torch-free `PapageiEncoderWeights` so serving stays NumPy-only.

torch is a guarded import: this module loads without it and `fine_tune_papagei` raises
`TrainBackendUnavailable` if it is missing — mirroring `mlx_encoder`.

The reported held-out MAE is computed via the NumPy serving path (`predict_hr`) on the
exported weights — the honest production number, not the torch-side training loss.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np

from ai.training.backends import TrainBackendUnavailable
from ai.training.config import DEFAULT_SEED, set_global_seed
from ai.training.mlx_encoder import EpochLog, ProgressCallback, TrainingHistory
from ai.training.papagei_resnet import PapageiEncoderWeights, znorm_segments

DEFAULT_PAPAGEI_CHECKPOINT = "models/cache/papagei/papagei_s.pt"


@dataclass(frozen=True)
class PapageiFineTuneConfig:
    """Fine-tuning hyperparameters. Defaults are a full-quality run (all data, best-val
    checkpoint kept) — a pretrained trunk converges in fewer epochs than from-scratch,
    so 60 is a generous budget for it, NOT a speed cap. `learning_rate` is deliberately
    small (pretrained trunk); the fresh head trains at `head_lr_mult`× that."""

    seed: int = DEFAULT_SEED
    epochs: int = 60
    learning_rate: float = 1e-4  # trunk LR (top of warmup→cosine)
    head_lr_mult: float = 10.0  # fresh HR head learns faster than the pretrained trunk
    batch_size: int = 64
    val_fraction: float = 0.25
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05


def _select_device(prefer_mps: bool = True) -> object:
    import torch

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * peak * (1.0 + np.cos(np.pi * min(1.0, progress)))


def fine_tune_papagei(
    signals_train: np.ndarray,
    y_train: np.ndarray,
    signals_val: np.ndarray,
    y_val: np.ndarray,
    *,
    sample_rate_hz: float,
    window_samples: int,
    pretrained_checkpoint: str = DEFAULT_PAPAGEI_CHECKPOINT,
    config: PapageiFineTuneConfig | None = None,
    on_epoch: ProgressCallback | None = None,
) -> tuple[PapageiEncoderWeights, TrainingHistory]:
    """Full-fine-tune the pretrained PaPaGei-S trunk + a fresh HR head on MPS/CPU.

    Returns exported NumPy `PapageiEncoderWeights` (best-val checkpoint) + per-epoch
    history. The exported trunk is validated against the torch model by the caller's
    post-fine-tune parity check.
    """
    cfg = config or PapageiFineTuneConfig()
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - guarded like the MLX backend
        raise TrainBackendUnavailable(
            "PyTorch is required to fine-tune PaPaGei-S (pip install -r requirements-papagei.txt)"
        ) from exc

    from ai.training._papagei_reference_torch import build_papagei_s, load_papagei_s_state_dict
    from ai.training.convert_papagei_weights import build_encoder_weights

    set_global_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _select_device()

    # Pretrained trunk + fresh HR head.
    model = build_papagei_s()
    model = load_papagei_s_state_dict(model, pretrained_checkpoint)
    model = model.float().to(device)
    head = nn.Linear(model.dense.in_features, 1).float().to(device)

    # z-norm inputs (serving contract) and standardise HR targets on TRAIN stats.
    xtr = znorm_segments(signals_train).astype(np.float32)[:, np.newaxis, :]
    xva = znorm_segments(signals_val).astype(np.float32)[:, np.newaxis, :]
    hr_mean, hr_std = float(np.mean(y_train)), float(np.std(y_train))
    hr_std = hr_std if hr_std > 1e-6 else 1.0
    ytr_std = ((y_train - hr_mean) / hr_std).astype(np.float32)

    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr_std).to(device)
    xva_t = torch.from_numpy(xva).to(device)

    params = [
        {"params": model.parameters(), "lr": cfg.learning_rate},
        {"params": head.parameters(), "lr": cfg.learning_rate * cfg.head_lr_mult},
    ]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    n = xtr_t.shape[0]
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = int(cfg.warmup_frac * total_steps)
    rng = np.random.default_rng(cfg.seed)

    logs: list[EpochLog] = []
    best_val_mae = float("inf")
    best_state: dict[str, object] | None = None
    best_epoch = 0
    step = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        head.train()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        last_lr = cfg.learning_rate
        for b in range(0, n, cfg.batch_size):
            idx = perm[b : b + cfg.batch_size]
            xb = xtr_t[idx]
            yb = ytr_t[idx]
            lr = _lr_at(step, total_steps, cfg.learning_rate, warmup)
            opt.param_groups[0]["lr"] = lr
            opt.param_groups[1]["lr"] = lr * cfg.head_lr_mult
            last_lr = lr
            opt.zero_grad()
            emb = model.embedding(xb)
            pred = head(emb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu()) * len(idx)
            step += 1
        train_mse = epoch_loss / n

        # Validation via torch (float32) — the final reported MAE uses the NumPy path.
        model.eval()
        head.eval()
        with torch.no_grad():
            pred_std = head(model.embedding(xva_t)).squeeze(-1).cpu().numpy()
        val_pred = pred_std * hr_std + hr_mean
        err = val_pred - y_val
        val_mae = float(np.mean(np.abs(err)))
        val_rmse = float(np.sqrt(np.mean(err**2)))

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "head": copy.deepcopy(head.state_dict()),
            }
        log = EpochLog(
            epoch=epoch, epochs_total=cfg.epochs, train_mse=train_mse,
            val_mae_bpm=val_mae, val_rmse_bpm=val_rmse, learning_rate=last_lr,
            is_best=is_best, seconds=time.time() - t0,
        )
        logs.append(log)
        if on_epoch is not None:
            on_epoch(log)

    assert best_state is not None
    model.load_state_dict(best_state["model"])  # type: ignore[arg-type]
    head.load_state_dict(best_state["head"])  # type: ignore[arg-type]

    # Export the BEST trunk + head to float64 NumPy (serving container). Running BN in
    # float64 on CPU makes the export parity-checkable to machine precision.
    model = model.cpu().double().eval()
    head = head.cpu().double().eval()
    head_w = head.weight.detach().cpu().double().numpy().reshape(-1)  # [512]
    head_b = float(head.bias.detach().cpu().double().numpy()[0])
    weights = build_encoder_weights(
        model, head_w=head_w, head_b=head_b, hr_mean=hr_mean, hr_std=hr_std,
        sample_rate_hz=sample_rate_hz, window_samples=window_samples,
    )

    # SHIPPED-MODEL parity: the exported NumPy trunk must reproduce the fine-tuned torch
    # trunk to ~machine precision (float64). A mismatch means a broken export — fail loud
    # rather than silently serve a wrong model.
    from ai.training.papagei_resnet import papagei_trunk_forward

    chk = znorm_segments(signals_val[: min(16, len(signals_val))]).astype(np.float64)
    chk = chk[:, np.newaxis, :]
    with torch.no_grad():
        emb_torch = model.embedding(torch.from_numpy(chk)).cpu().numpy()
    emb_numpy = papagei_trunk_forward(weights, chk)
    max_abs = float(np.abs(emb_torch - emb_numpy).max())
    if not np.isfinite(max_abs) or max_abs > 1e-7:
        raise RuntimeError(
            f"post-fine-tune export parity failed: max|Δ|={max_abs:.3e} (>1e-7) — "
            "the NumPy serving trunk does not match the fine-tuned torch trunk"
        )
    return weights, TrainingHistory(logs=tuple(logs), best_epoch=best_epoch)
