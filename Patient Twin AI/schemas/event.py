"""Event contract (docs/02 §4, docs/04 §3, docs/05 §6).

`EventCandidate` is what the Event Engine emits — a multi-metric candidate event
built from co-occurring deviations. The Patient State Engine maps it to a versioned,
append-only `EventNode` (schemas.psg) when it commits it to the PSG. Events are
advisory inputs to the LLM and Policy Engine; they are NEVER surfaced to the patient
directly (docs/05 §6).

CLAUDE.md: contracts live in schemas/ only.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from .psg import EventSeverity


class EventStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"


class EventCandidate(BaseModel):
    """A candidate event proposed by the Event Engine (docs/05 §6)."""

    patient_id: UUID
    type: str  # clinical event type from the (versioned) rule — advisory, never a diagnosis
    severity: EventSeverity
    status: EventStatus = EventStatus.ACTIVE
    onset_ts: datetime
    contributing_deviation_ids: list[UUID] = Field(min_length=1)
    rule_id: str  # which versioned co-occurrence rule fired

    @model_validator(mode="after")
    def _onset_tz_aware(self) -> EventCandidate:
        if self.onset_ts.tzinfo is None:
            raise ValueError("onset_ts must be timezone-aware — naive datetimes are rejected")
        return self
