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

from collections.abc import Callable
from typing import Protocol, TypeVar
from uuid import UUID

from schemas.psg import (
    AllergyNode,
    BaselineNode,
    ConditionNode,
    DeviationNode,
    DocumentNode,
    EventNode,
    ForecastNode,
    MedicationNode,
    ObservationNode,
    VersionedNode,
)

_N = TypeVar("_N", bound=VersionedNode)


def _latest_by_key(nodes: list[_N], patient_id: UUID, key: Callable[[_N], str]) -> list[_N]:
    """Current node per key for one patient = highest-version row per key."""
    latest: dict[str, _N] = {}
    for node in nodes:
        if node.patient_id != patient_id:
            continue
        k = key(node)
        if k not in latest or node.version > latest[k].version:
            latest[k] = node
    return list(latest.values())


class PSGStore(Protocol):
    def add_baseline(self, node: BaselineNode) -> None: ...
    def add_deviation(self, node: DeviationNode) -> None: ...
    def add_event(self, node: EventNode) -> None: ...
    def add_forecast(self, node: ForecastNode) -> None: ...
    def add_document(self, node: DocumentNode) -> None: ...
    def add_condition(self, node: ConditionNode) -> None: ...
    def add_medication(self, node: MedicationNode) -> None: ...
    def add_allergy(self, node: AllergyNode) -> None: ...
    def add_observation(self, node: ObservationNode) -> None: ...
    def current_baseline(
        self, patient_id: UUID, metric_code: str, context: str
    ) -> BaselineNode | None: ...
    def current_baselines(self, patient_id: UUID) -> list[BaselineNode]: ...
    def recent_deviations(self, patient_id: UUID, *, limit: int) -> list[DeviationNode]: ...
    def active_events(self, patient_id: UUID) -> list[EventNode]: ...
    def latest_forecasts(self, patient_id: UUID) -> list[ForecastNode]: ...
    def recent_documents(self, patient_id: UUID, *, limit: int) -> list[DocumentNode]: ...
    def current_conditions(self, patient_id: UUID) -> list[ConditionNode]: ...
    def current_medications(self, patient_id: UUID) -> list[MedicationNode]: ...
    def current_allergies(self, patient_id: UUID) -> list[AllergyNode]: ...
    def recent_observations(self, patient_id: UUID, *, limit: int) -> list[ObservationNode]: ...


class InMemoryPSGStore:
    """In-memory dev/test store. Keeps only the current version per key (baselines);
    deviations accumulate. Full append-only history lives in `SqlAlchemyPSGStore`.
    """

    def __init__(self) -> None:
        self._baselines: list[BaselineNode] = []
        self._deviations: list[DeviationNode] = []
        self._events: list[EventNode] = []
        self._forecasts: list[ForecastNode] = []
        self._documents: list[DocumentNode] = []
        self._conditions: list[ConditionNode] = []
        self._medications: list[MedicationNode] = []
        self._allergies: list[AllergyNode] = []
        self._observations: list[ObservationNode] = []

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

    def add_event(self, node: EventNode) -> None:
        self._events.append(node)

    def add_forecast(self, node: ForecastNode) -> None:
        self._forecasts.append(node)

    def add_document(self, node: DocumentNode) -> None:
        self._documents.append(node)

    def add_condition(self, node: ConditionNode) -> None:
        self._conditions.append(node)

    def add_medication(self, node: MedicationNode) -> None:
        self._medications.append(node)

    def add_allergy(self, node: AllergyNode) -> None:
        self._allergies.append(node)

    def add_observation(self, node: ObservationNode) -> None:
        self._observations.append(node)

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

    def active_events(self, patient_id: UUID) -> list[EventNode]:
        events = [e for e in self._events if e.patient_id == patient_id and e.status == "active"]
        events.sort(key=lambda e: (e.onset_ts, e.id.int), reverse=True)
        return events

    def latest_forecasts(self, patient_id: UUID) -> list[ForecastNode]:
        latest: dict[str, ForecastNode] = {}
        for f in self._forecasts:
            if f.patient_id != patient_id:
                continue
            key = f.metric_code.value
            if key not in latest or f.generated_at > latest[key].generated_at:
                latest[key] = f
        return list(latest.values())

    def recent_documents(self, patient_id: UUID, *, limit: int) -> list[DocumentNode]:
        docs = [d for d in self._documents if d.patient_id == patient_id]
        docs.sort(key=lambda d: (d.created_at, d.id.int), reverse=True)
        return docs[:limit]

    def current_conditions(self, patient_id: UUID) -> list[ConditionNode]:
        return _latest_by_key(self._conditions, patient_id, lambda n: n.snomed_code)

    def current_medications(self, patient_id: UUID) -> list[MedicationNode]:
        return _latest_by_key(self._medications, patient_id, lambda n: n.rxnorm_code)

    def current_allergies(self, patient_id: UUID) -> list[AllergyNode]:
        return _latest_by_key(self._allergies, patient_id, lambda n: n.substance_code)

    def recent_observations(self, patient_id: UUID, *, limit: int) -> list[ObservationNode]:
        obs = [o for o in self._observations if o.patient_id == patient_id]
        obs.sort(key=lambda o: (o.ts, o.id.int), reverse=True)
        return obs[:limit]
