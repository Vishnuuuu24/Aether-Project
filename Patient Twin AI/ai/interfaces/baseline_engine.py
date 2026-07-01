"""Stable `BaselineEngine` interface (docs/02 §6, docs/05 §2).

v1 impl: `ai.baseline.StatisticalBaselineEngine`.
DEFERRED impl: `FoundationEncoderBaselineEngine` (PaPaGei-S / Pulse-PPG embeddings ->
per-user density) — a new implementation of THIS protocol, never a new call site.

The protocol is per-patient scoped: `get_baseline` carries no patient_id, so an
engine instance owns exactly one patient's state (docs/05 §2 signatures).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas.baseline import Baseline, DeviationResult
from schemas.reading import Reading


@runtime_checkable
class BaselineEngine(Protocol):
    def update(self, reading: Reading) -> None: ...
    def score(self, reading: Reading) -> DeviationResult: ...
    def get_baseline(self, metric_code: str, context: str) -> Baseline: ...
