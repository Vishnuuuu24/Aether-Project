"""Where accepted, normalised readings are handed off to the SQI stage.

Ingestion must not compute SQI/features or write the PSG (docs/02 §2); it emits
readings to a sink that the SQI service consumes. `InMemoryReadingSink` is for
tests/dev; the production sink enqueues to Redis (docs/08 wires `REDIS_URL` on the
ingestion service) for the SQI + feature service to pick up.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from schemas.reading import Reading


class ReadingSink(Protocol):
    def emit(self, readings: Sequence[Reading]) -> None: ...


class InMemoryReadingSink:
    def __init__(self) -> None:
        self.readings: list[Reading] = []

    def emit(self, readings: Sequence[Reading]) -> None:
        self.readings.extend(readings)
