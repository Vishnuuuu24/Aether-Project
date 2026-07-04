"""Hash-chained audit record. docs/04 §7. Append-only; core/audit owns the
hashing logic — this module only defines the shape.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AuditActor(str, Enum):
    SYSTEM = "system"
    PATIENT = "patient"
    CLINICIAN = "clinician"


class AuditAction(str, Enum):
    INGEST = "ingest"
    SQI = "sqi"
    BASELINE_UPDATE = "baseline_update"
    STATE_COMMIT = "state_commit"
    RETRIEVE = "retrieve"
    LLM_CALL = "llm_call"
    POLICY_DECISION = "policy_decision"
    OUTPUT = "output"
    CONSENT_CHANGE = "consent_change"
    OUTCOME_CAPTURE = "outcome_capture"
    ESCALATION_ACK = "escalation_ack"


class AuditRecord(BaseModel):
    audit_id: UUID = Field(default_factory=uuid4)
    patient_id: UUID
    actor: AuditActor
    action: AuditAction
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    versions: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime
    prev_hash: str
    hash: str
