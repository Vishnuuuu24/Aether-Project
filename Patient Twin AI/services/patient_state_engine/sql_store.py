"""Postgres-backed PSG store — the relational realisation of the PSG (docs/04 §3).

Converts between the pydantic node contracts (`schemas.psg`) and the ORM rows
(`core.db.models`). Append-only: `add_*` only inserts; "current" is the highest
`version` per key (equivalent to "not referenced by any supersedes" for a linear
chain). Like `SqlAlchemyAuditStore`, this store does NOT commit — the caller owns
the transaction so a node write and its audit record commit atomically.

Concurrency: v1 assumes a single writer per patient (per-patient serial processing).
Under true concurrent writers the read-current -> insert-next-version step could fork
into two same-version rows; a `UNIQUE(patient_id, metric_code, context, version)`
constraint is the hardening fix (deferred — needs a migration).
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from schemas.psg import BaselineNode as BaselineNodeSchema
from schemas.psg import DeviationDirection
from schemas.psg import DeviationNode as DeviationNodeSchema
from schemas.reading import MeasurementContext, MetricCode


class SqlAlchemyPSGStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_baseline(self, node: BaselineNodeSchema) -> None:
        from core.db.models import BaselineNode as BaselineRow

        self._session.add(
            BaselineRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                metric_code=node.metric_code.value,
                context=node.context.value,
                method=node.method,
                center=node.center,
                dispersion=node.dispersion,
                sample_n=node.sample_n,
                window_spec=node.window_spec,
                confidence=node.confidence,
                is_population_fallback=node.is_population_fallback,
            )
        )
        self._session.flush()

    def add_deviation(self, node: DeviationNodeSchema) -> None:
        from core.db.models import DeviationNode as DeviationRow

        self._session.add(
            DeviationRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                metric_code=node.metric_code.value,
                baseline_id=node.baseline_id,
                magnitude=node.magnitude,
                direction=node.direction.value,
                z_robust=node.z_robust,
                confidence=node.confidence,
                is_population_fallback=node.is_population_fallback,
            )
        )
        self._session.flush()

    def current_baseline(
        self, patient_id: UUID, metric_code: str, context: str
    ) -> BaselineNodeSchema | None:
        from core.db.models import BaselineNode as BaselineRow

        row = self._session.execute(
            select(BaselineRow)
            .where(
                BaselineRow.patient_id == patient_id,
                BaselineRow.metric_code == metric_code,
                BaselineRow.context == context,
            )
            .order_by(BaselineRow.version.desc())
            .limit(1)
        ).scalar_one_or_none()
        return _baseline_to_schema(row) if row is not None else None

    def current_baselines(self, patient_id: UUID) -> list[BaselineNodeSchema]:
        from core.db.models import BaselineNode as BaselineRow

        rows = (
            self._session.execute(
                select(BaselineRow)
                .where(BaselineRow.patient_id == patient_id)
                .order_by(BaselineRow.version.asc())
            )
            .scalars()
            .all()
        )
        latest: dict[tuple[str, str], BaselineRow] = {}
        for row in rows:  # ascending version => last write per key wins
            latest[(row.metric_code, row.context)] = row
        return [_baseline_to_schema(row) for row in latest.values()]

    def recent_deviations(self, patient_id: UUID, *, limit: int) -> list[DeviationNodeSchema]:
        from core.db.models import DeviationNode as DeviationRow

        rows = (
            self._session.execute(
                select(DeviationRow)
                .where(DeviationRow.patient_id == patient_id)
                .order_by(DeviationRow.created_at.desc(), DeviationRow.id.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [_deviation_to_schema(row) for row in rows]


def _baseline_to_schema(r: Any) -> BaselineNodeSchema:
    return BaselineNodeSchema(
        id=r.id,
        patient_id=r.patient_id,
        version=r.version,
        supersedes=r.supersedes,
        created_at=r.created_at,
        created_by=r.created_by,
        metric_code=MetricCode(r.metric_code),
        context=MeasurementContext(r.context),
        method=r.method,
        center=r.center,
        dispersion=r.dispersion,
        sample_n=r.sample_n,
        window_spec=r.window_spec,
        confidence=r.confidence,
        is_population_fallback=r.is_population_fallback,
    )


def _deviation_to_schema(r: Any) -> DeviationNodeSchema:
    return DeviationNodeSchema(
        id=r.id,
        patient_id=r.patient_id,
        version=r.version,
        supersedes=r.supersedes,
        created_at=r.created_at,
        created_by=r.created_by,
        metric_code=MetricCode(r.metric_code),
        baseline_id=r.baseline_id,
        magnitude=r.magnitude,
        direction=DeviationDirection(r.direction),
        z_robust=r.z_robust,
        confidence=r.confidence,
        is_population_fallback=r.is_population_fallback,
    )
