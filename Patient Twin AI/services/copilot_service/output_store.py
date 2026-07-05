"""Postgres-backed OutputStore (docs/04 §6; docs/15 T7.2c).

Persists every user-facing `OutputContract` — approved, downgraded, abstained, or
suppressed — as an `output` row, so any answer the system gave is reconstructable
(alongside its `POLICY_DECISION` audit record). Implements the copilot's `OutputStore`
port; does NOT commit — the caller owns the request transaction.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from schemas.output_contract import OutputContract


class SqlOutputStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, output: OutputContract) -> None:
        from core.db.models import OutputRecord

        baseline_ref = (
            output.baseline_reference.model_dump(mode="json")
            if output.baseline_reference is not None
            else None
        )
        self._session.add(
            OutputRecord(
                output_id=output.output_id,
                patient_id=output.patient_id,
                type=output.type.value,
                message=output.message,
                severity=output.severity.value,
                confidence=output.confidence,
                evidence=[e.model_dump(mode="json") for e in output.evidence],
                baseline_reference=baseline_ref,
                recommended_action=output.recommended_action.value,
                escalation=output.escalation.model_dump(mode="json"),
                abstained=output.abstained.model_dump(mode="json"),
                policy=output.policy.model_dump(mode="json"),
                disclaimer=output.disclaimer,
                versions=output.versions.model_dump(mode="json"),
                created_at=output.created_at,
            )
        )
        self._session.flush()
