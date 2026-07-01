"""Ingestion orchestration (docs/02 §2): normalise → consent-gate → hand off to
the SQI stage, emitting an audit event per patient.

Boundary (do NOT cross): no SQI/feature computation, no baseline, no PSG write, no
LLM. Consent is deny-by-default and applies to every actor including `system`
(CLAUDE.md) — a reading whose patient has not consented to `vitals` is rejected,
not silently processed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any
from uuid import UUID

from core.audit import AuditWriter
from core.auth.consent_gate import consent_covers
from schemas.audit import AuditAction, AuditActor
from schemas.consent import ConsentScope
from schemas.reading import IngestBatchResult, IngestRejection, Reading

from .consent import ConsentProvider
from .normaliser import normalise_batch
from .sink import ReadingSink


class IngestionService:
    def __init__(
        self,
        *,
        consent_provider: ConsentProvider,
        sink: ReadingSink,
        audit_writer: AuditWriter | None = None,
        actor: AuditActor = AuditActor.SYSTEM,
    ) -> None:
        self._consent = consent_provider
        self._sink = sink
        self._audit = audit_writer
        self._actor = actor

    def ingest(self, items: Iterable[Mapping[str, Any]], *, adapter: str) -> IngestBatchResult:
        norm = normalise_batch(items, default_adapter=adapter)
        rejections = list(norm.rejections)

        accepted: list[Reading] = []
        for index, reading in norm.accepted:
            consent = self._consent.get_consent(reading.patient_id)
            if consent_covers(consent, ConsentScope.VITALS):
                accepted.append(reading)
            else:
                rejections.append(
                    {
                        "index": index,
                        "errors": [
                            IngestRejection(
                                field="consent",
                                issue=(
                                    f"patient {reading.patient_id} has not consented to 'vitals'"
                                ),
                            ).model_dump()
                        ],
                    }
                )

        if accepted:
            self._sink.emit(accepted)
            self._emit_audit(accepted, adapter=adapter)

        return IngestBatchResult(accepted=[r.reading_id for r in accepted], rejected=rejections)

    def _emit_audit(self, readings: list[Reading], *, adapter: str) -> None:
        if self._audit is None:
            return
        by_patient: dict[UUID, list[Reading]] = defaultdict(list)
        for reading in readings:
            by_patient[reading.patient_id].append(reading)
        for patient_id, patient_readings in by_patient.items():
            self._audit.write(
                patient_id=patient_id,
                actor=self._actor,
                action=AuditAction.INGEST,
                input_refs=[f"adapter:{adapter}"],
                output_refs=[f"reading:{r.reading_id}" for r in patient_readings],
            )
