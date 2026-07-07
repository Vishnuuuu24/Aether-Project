"""Stress-context task head on the frozen biosignal-encoder embedding (docs/16 Sprint 10).

The Sprint 10 DoD asks the learned path to derive **HR *and* stress-context** from raw
PPG. HR is the conv encoder's regression head; stress-context is a second *task head* on
the SAME 128-d encoder embedding — a plain logistic regression trained in NumPy (like the
Sprint 9 `TrainedHead`), so it needs no MLX and rides the identical NumPy serving path.

Principle 2 holds: the raw waveform is consumed only to produce the embedding; the head
sees the structured embedding, never the signal. The head is a NEW output on the existing
`FeatureExtractor` seam (`stress_probability`), never a new call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ai.training.encoder_model import EncoderWeights, encoder_embedding

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

STRESS_HEAD_VERSION = "ppg-stress-logreg-v1"


@dataclass(frozen=True)
class StressHead:
    """A logistic-regression stress-context head over the encoder embedding.

    `feat_mean`/`feat_std` standardise the embedding (fit on train); `w`/`b` are the
    logistic weights. `predict_proba` returns P(stress). Backend-agnostic NumPy.
    """

    w: FloatArray  # [EMBEDDING_DIM]
    b: float
    feat_mean: FloatArray  # [EMBEDDING_DIM]
    feat_std: FloatArray  # [EMBEDDING_DIM]
    version: str = STRESS_HEAD_VERSION


def _sigmoid(z: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def embed_windows(weights: EncoderWeights, signals: FloatArray) -> FloatArray:
    """Frozen-trunk embedding [N, EMBEDDING_DIM] for raw windows [N, L]."""
    return encoder_embedding(weights, np.asarray(signals, dtype=np.float64))


def predict_stress_proba(head: StressHead, embeddings: FloatArray) -> FloatArray:
    """P(stress) for [N, EMBEDDING_DIM] embeddings."""
    z = (np.asarray(embeddings, dtype=np.float64) - head.feat_mean) / head.feat_std
    return _sigmoid(z @ head.w + head.b)


def train_stress_head(
    embeddings: FloatArray,
    labels: FloatArray,
    *,
    epochs: int = 400,
    learning_rate: float = 0.1,
    l2: float = 1e-3,
    seed: int = 0,
) -> StressHead:
    """Fit a logistic head on frozen embeddings by full-batch gradient descent.

    Standardises the embedding (train stats), then minimises L2-regularised logistic
    loss. Class-balanced sample weights keep an uneven baseline/stress split honest.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("embeddings [N, D] and labels [N] must align")
    if x.shape[0] < 2 or len(np.unique(y)) < 2:
        raise ValueError("need at least two windows of each class to fit the stress head")

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xn = (x - mean) / std

    # Class-balanced weights: each class contributes equally regardless of count.
    n = y.size
    pos = float(y.sum())
    neg = float(n - pos)
    sw = np.where(y > 0.5, n / (2.0 * max(pos, 1.0)), n / (2.0 * max(neg, 1.0)))

    rng = np.random.default_rng(seed)
    d = xn.shape[1]
    w = rng.normal(0.0, 0.01, size=d)
    b = 0.0
    sw_sum = float(sw.sum())
    for _ in range(epochs):
        p = _sigmoid(xn @ w + b)
        g = sw * (p - y)
        grad_w = xn.T @ g / sw_sum + l2 * w
        grad_b = float(g.sum() / sw_sum)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
    return StressHead(w=w, b=b, feat_mean=mean, feat_std=std)
