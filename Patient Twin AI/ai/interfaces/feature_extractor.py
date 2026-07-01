"""Stable `FeatureExtractor` interface (docs/02 §6).

v1 impl: `ai.features.ClassicalFeatureExtractor` (+SQI gate).
DEFERRED impl: biosignal foundation encoder (PaPaGei-S / Pulse-PPG) — a new
implementation of THIS protocol, never a new call site (docs/05 §3).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas.features import FeatureSet, SignalWindow


@runtime_checkable
class FeatureExtractor(Protocol):
    def extract(self, window: SignalWindow) -> FeatureSet: ...
