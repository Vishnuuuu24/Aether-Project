"""Clinical outcome capture for outer-loop learning. docs/11 §3, docs/07 §7.

The system records real clinical outcomes (admission, diagnosis, medication
change) linked to the *outputs and PSG/engine versions that preceded them*. This
is the labelled signal for later human-gated retraining. v1 captures and stores
only — nothing here feeds back into a live model (CLAUDE.md principle 5: no
closed-loop self-modification).

Like every other contract, this shape is defined once here and imported by the
governance service; no service redefines it.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class OutcomeType(str, Enum):
    ADMISSION = "admission"
    DIAGNOSIS = "diagnosis"
    MEDICATION_CHANGE = "medication_change"
    PROCEDURE = "procedure"
    DEATH = "death"
    OTHER = "other"


class OutcomeSource(str, Enum):
    """Who asserted the outcome. Never the LLM — outcomes are ground truth."""

    CLINICIAN = "clinician"
    EHR_IMPORT = "ehr_import"
    PATIENT_REPORTED = "patient_reported"


class Outcome(BaseModel):
    """A recorded real-world clinical outcome, linked to prior system outputs.

    `linked_output_ids` are the `OutputContract.output_id`s this outcome is
    evaluated against; `versions` snapshots the model/ruleset/baseline-engine
    versions in force when those outputs were produced, so a later retraining run
    can join outcomes to the exact artefacts that preceded them.
    """

    outcome_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID
    outcome_type: OutcomeType
    occurred_at: datetime
    detail: str = Field(min_length=1)
    code: str | None = None  # optional coded value (ICD-10 / SNOMED), when known
    linked_output_ids: list[UUID] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    source: OutcomeSource
    recorded_at: datetime

    @model_validator(mode="after")
    def _timezone_aware(self) -> Outcome:
        for name, value in (("occurred_at", self.occurred_at), ("recorded_at", self.recorded_at)):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        return self
