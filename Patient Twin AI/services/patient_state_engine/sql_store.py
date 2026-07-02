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

from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from schemas.psg import AllergyNode as AllergyNodeSchema
from schemas.psg import BaselineNode as BaselineNodeSchema
from schemas.psg import ConditionNode as ConditionNodeSchema
from schemas.psg import DeviationDirection, EventSeverity
from schemas.psg import DeviationNode as DeviationNodeSchema
from schemas.psg import DocumentNode as DocumentNodeSchema
from schemas.psg import EventNode as EventNodeSchema
from schemas.psg import ForecastNode as ForecastNodeSchema
from schemas.psg import MedicationNode as MedicationNodeSchema
from schemas.psg import ObservationNode as ObservationNodeSchema
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

    def add_event(self, node: EventNodeSchema) -> None:
        from core.db.models import EventNode as EventRow

        self._session.add(
            EventRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                type=node.type,
                severity=node.severity.value,
                status=node.status,
                onset_ts=node.onset_ts,
                contributing_deviation_ids=list(node.contributing_deviation_ids),
            )
        )
        self._session.flush()

    def active_events(self, patient_id: UUID) -> list[EventNodeSchema]:
        from core.db.models import EventNode as EventRow

        rows = (
            self._session.execute(
                select(EventRow)
                .where(EventRow.patient_id == patient_id, EventRow.status == "active")
                .order_by(EventRow.onset_ts.desc(), EventRow.id.desc())
            )
            .scalars()
            .all()
        )
        return [_event_to_schema(row) for row in rows]

    def add_forecast(self, node: ForecastNodeSchema) -> None:
        from core.db.models import ForecastNode as ForecastRow

        self._session.add(
            ForecastRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                metric_code=node.metric_code.value,
                horizon_days=node.horizon_days,
                points=list(node.points),
                intervals=[list(iv) for iv in node.intervals],
                method=node.method,
                generated_at=node.generated_at,
            )
        )
        self._session.flush()

    def latest_forecasts(self, patient_id: UUID) -> list[ForecastNodeSchema]:
        from core.db.models import ForecastNode as ForecastRow

        rows = (
            self._session.execute(
                select(ForecastRow)
                .where(ForecastRow.patient_id == patient_id)
                .order_by(ForecastRow.generated_at.asc())
            )
            .scalars()
            .all()
        )
        latest: dict[str, Any] = {}
        for row in rows:  # ascending generated_at => last per metric wins
            latest[row.metric_code] = row
        return [_forecast_to_schema(row) for row in latest.values()]

    def add_document(self, node: DocumentNodeSchema) -> None:
        from core.db.models import DocumentNode as DocumentRow

        self._session.add(
            DocumentRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                doc_type=node.doc_type,
                uri=node.uri,
                ocr_ref=node.ocr_ref,
                codes=list(node.codes),
            )
        )
        self._session.flush()

    def add_condition(self, node: ConditionNodeSchema) -> None:
        from core.db.models import ConditionNode as ConditionRow

        self._session.add(
            ConditionRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                snomed_code=node.snomed_code,
                display=node.display,
                status=node.status,
                onset=node.onset,
                source_document_id=node.source_document_id,
            )
        )
        self._session.flush()

    def add_medication(self, node: MedicationNodeSchema) -> None:
        from core.db.models import MedicationNode as MedicationRow

        self._session.add(
            MedicationRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                rxnorm_code=node.rxnorm_code,
                display=node.display,
                dose=node.dose,
                status=node.status,
                source_document_id=node.source_document_id,
            )
        )
        self._session.flush()

    def add_allergy(self, node: AllergyNodeSchema) -> None:
        from core.db.models import AllergyNode as AllergyRow

        self._session.add(
            AllergyRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                substance_code=node.substance_code,
                reaction=node.reaction,
                severity=node.severity,
                source=node.source,
                status=node.status,
            )
        )
        self._session.flush()

    def add_observation(self, node: ObservationNodeSchema) -> None:
        from core.db.models import ObservationNode as ObservationRow

        self._session.add(
            ObservationRow(
                id=node.id,
                patient_id=node.patient_id,
                version=node.version,
                supersedes=node.supersedes,
                created_at=node.created_at,
                created_by=node.created_by,
                loinc_code=node.loinc_code,
                display=node.display,
                value=node.value,
                unit=node.unit,
                ts=node.ts,
                source_document_id=node.source_document_id,
                status=node.status,
            )
        )
        self._session.flush()

    def current_conditions(self, patient_id: UUID) -> list[ConditionNodeSchema]:
        from core.db.models import ConditionNode as ConditionRow

        rows = self._latest_by_key(ConditionRow, patient_id, lambda r: r.snomed_code)
        return [_condition_to_schema(r) for r in rows]

    def current_medications(self, patient_id: UUID) -> list[MedicationNodeSchema]:
        from core.db.models import MedicationNode as MedicationRow

        rows = self._latest_by_key(MedicationRow, patient_id, lambda r: r.rxnorm_code)
        return [_medication_to_schema(r) for r in rows]

    def current_allergies(self, patient_id: UUID) -> list[AllergyNodeSchema]:
        from core.db.models import AllergyNode as AllergyRow

        rows = self._latest_by_key(AllergyRow, patient_id, lambda r: r.substance_code)
        return [_allergy_to_schema(r) for r in rows]

    def recent_observations(self, patient_id: UUID, *, limit: int) -> list[ObservationNodeSchema]:
        from core.db.models import ObservationNode as ObservationRow

        rows = (
            self._session.execute(
                select(ObservationRow)
                .where(ObservationRow.patient_id == patient_id)
                .order_by(ObservationRow.ts.desc(), ObservationRow.id.desc())
                .limit(limit)
            )
            .scalars()
            .all()
        )
        return [_observation_to_schema(r) for r in rows]

    def _latest_by_key(
        self, row_cls: Any, patient_id: UUID, key: Callable[[Any], str]
    ) -> list[Any]:
        rows = (
            self._session.execute(
                select(row_cls)
                .where(row_cls.patient_id == patient_id)
                .order_by(row_cls.version.asc())
            )
            .scalars()
            .all()
        )
        latest: dict[str, Any] = {}
        for row in rows:  # ascending version => last write per key wins
            latest[key(row)] = row
        return list(latest.values())

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


def _condition_to_schema(r: Any) -> ConditionNodeSchema:
    return ConditionNodeSchema(
        id=r.id,
        patient_id=r.patient_id,
        version=r.version,
        supersedes=r.supersedes,
        created_at=r.created_at,
        created_by=r.created_by,
        snomed_code=r.snomed_code,
        display=r.display,
        status=r.status,
        onset=r.onset,
        source_document_id=r.source_document_id,
    )


def _medication_to_schema(r: Any) -> MedicationNodeSchema:
    return MedicationNodeSchema(
        id=r.id,
        patient_id=r.patient_id,
        version=r.version,
        supersedes=r.supersedes,
        created_at=r.created_at,
        created_by=r.created_by,
        rxnorm_code=r.rxnorm_code,
        display=r.display,
        dose=r.dose,
        status=r.status,
        source_document_id=r.source_document_id,
    )


def _allergy_to_schema(r: Any) -> AllergyNodeSchema:
    return AllergyNodeSchema(
        id=r.id,
        patient_id=r.patient_id,
        version=r.version,
        supersedes=r.supersedes,
        created_at=r.created_at,
        created_by=r.created_by,
        substance_code=r.substance_code,
        reaction=r.reaction,
        severity=r.severity,
        source=r.source,
        status=r.status,
    )


def _observation_to_schema(r: Any) -> ObservationNodeSchema:
    return ObservationNodeSchema(
        id=r.id,
        patient_id=r.patient_id,
        version=r.version,
        supersedes=r.supersedes,
        created_at=r.created_at,
        created_by=r.created_by,
        loinc_code=r.loinc_code,
        display=r.display,
        value=r.value,
        unit=r.unit,
        ts=r.ts,
        source_document_id=r.source_document_id,
        status=r.status,
    )


def _forecast_to_schema(r: Any) -> ForecastNodeSchema:
    return ForecastNodeSchema(
        id=r.id,
        patient_id=r.patient_id,
        version=r.version,
        supersedes=r.supersedes,
        created_at=r.created_at,
        created_by=r.created_by,
        metric_code=MetricCode(r.metric_code),
        horizon_days=r.horizon_days,
        points=list(r.points),
        intervals=[tuple(iv) for iv in r.intervals],
        method=r.method,
        generated_at=r.generated_at,
    )


def _event_to_schema(r: Any) -> EventNodeSchema:
    return EventNodeSchema(
        id=r.id,
        patient_id=r.patient_id,
        version=r.version,
        supersedes=r.supersedes,
        created_at=r.created_at,
        created_by=r.created_by,
        type=r.type,
        severity=EventSeverity(r.severity),
        status=r.status,
        onset_ts=r.onset_ts,
        contributing_deviation_ids=list(r.contributing_deviation_ids),
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
