"""Consent schema. docs/04 §1.

No processing proceeds without a valid, scoped, non-revoked consent record
covering the relevant scope. That enforcement lives in `core.auth.consent_gate`;
this module only defines the shape. The patient profile that embeds a consent
block lives in `schemas/patient.py`.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ConsentScope(str, Enum):
    VITALS = "vitals"
    DOCUMENTS = "documents"
    COPILOT = "copilot"
    FORECAST = "forecast"


class Consent(BaseModel):
    scope: list[ConsentScope]
    version: str
    granted_at: datetime
    revoked_at: datetime | None = None

    def covers(self, required: ConsentScope) -> bool:
        return self.revoked_at is None and required in self.scope
