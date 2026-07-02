"""Versioned PSG storage (docs/04 §3).

Each change writes a NEW version node with `supersedes` set to the prior node's id;
the "current" node for a key is the highest-version row. The production
`SqlAlchemyPSGStore` (sql_store.py) is fully append-only — every version is retained
as the audited history. `InMemoryPSGStore` is a lightweight dev/test store that
keeps only the current version per key (history retention is the SQL store's job);
its version/`supersedes` bookkeeping still matches so tests exercise the same paths.

The store operates on the pydantic node contracts (`schemas.psg`) so the engine is
storage-agnostic; the SQL store converts to/from the ORM rows.
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from schemas.psg import BaselineNode, DeviationNode


class PSGStore(Protocol):
    def add_baseline(self, node: BaselineNode) -> None: ...
    def add_deviation(self, node: DeviationNode) -> None: ...
    def current_baseline(
        self, patient_id: UUID, metric_code: str, context: str
    ) -> BaselineNode | None: ...
    def current_baselines(self, patient_id: UUID) -> list[BaselineNode]: ...
    def recent_deviations(self, patient_id: UUID, *, limit: int) -> list[DeviationNode]: ...


class InMemoryPSGStore:
    """In-memory dev/test store. Keeps only the current version per key (baselines);
    deviations accumulate. Full append-only history lives in `SqlAlchemyPSGStore`.
    """

    def __init__(self) -> None:
        self._baselines: list[BaselineNode] = []
        self._deviations: list[DeviationNode] = []

    def add_baseline(self, node: BaselineNode) -> None:
        self._baselines = [
            b
            for b in self._baselines
            if not (
                b.patient_id == node.patient_id
                and b.metric_code == node.metric_code
                and b.context == node.context
            )
        ]
        self._baselines.append(node)

    def add_deviation(self, node: DeviationNode) -> None:
        self._deviations.append(node)

    def current_baseline(
        self, patient_id: UUID, metric_code: str, context: str
    ) -> BaselineNode | None:
        candidates = [
            b
            for b in self._baselines
            if b.patient_id == patient_id
            and b.metric_code.value == metric_code
            and b.context.value == context
        ]
        return max(candidates, key=lambda b: b.version) if candidates else None

    def current_baselines(self, patient_id: UUID) -> list[BaselineNode]:
        latest: dict[tuple[str, str], BaselineNode] = {}
        for b in self._baselines:
            if b.patient_id != patient_id:
                continue
            key = (b.metric_code.value, b.context.value)
            if key not in latest or b.version > latest[key].version:
                latest[key] = b
        return list(latest.values())

    def recent_deviations(self, patient_id: UUID, *, limit: int) -> list[DeviationNode]:
        devs = [d for d in self._deviations if d.patient_id == patient_id]
        # id is a deterministic tiebreak on equal timestamps (matches the SQL store).
        devs.sort(key=lambda d: (d.created_at, d.id.int), reverse=True)
        return devs[:limit]
