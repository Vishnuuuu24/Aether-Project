"""Persisted user-facing outputs (docs/04 §6).

Every output — including abstentions and suppressions — is stored, so any
answer the system ever gave can be reconstructed and audited. The nested parts
of the contract (evidence, policy, versions, …) are kept as JSONB.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from ..base import Base


class OutputRecord(Base):
    __tablename__ = "output"

    output_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    baseline_reference: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    recommended_action: Mapped[str] = mapped_column(String, nullable=False)
    escalation: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    abstained: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    policy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    disclaimer: Mapped[str] = mapped_column(Text, nullable=False)
    versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
