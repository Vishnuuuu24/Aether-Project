"""Governance & compliance tables: patient profile, consent history, the
hash-chained audit log, and the version registry (docs/04 §1, §7; docs/06 §8).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from ..base import Base
from .common import NOW


class PatientProfile(Base):
    __tablename__ = "patient_profile"

    patient_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    sex_at_birth: Mapped[str] = mapped_column(String, nullable=False)
    dob: Mapped[date | None] = mapped_column(Date)
    age_years: Mapped[int | None] = mapped_column(Integer)
    gender: Mapped[str | None] = mapped_column(String)
    height_cm: Mapped[float | None] = mapped_column(Float)
    weight_kg: Mapped[float | None] = mapped_column(Float)
    weight_measured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    blood_group: Mapped[str | None] = mapped_column(String)
    physical_disability: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=NOW
    )


class Consent(Base):
    """Append-only consent history; current consent = latest non-revoked row."""

    __tablename__ = "consent"

    consent_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("patient_profile.patient_id"), nullable=False, index=True
    )
    scope: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    granted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=NOW
    )


class AuditLog(Base):
    """Single global hash chain (docs/04 §7). `seq` gives a total order; the chain
    links via prev_hash → hash. Append-only — never updated or deleted.
    """

    __tablename__ = "audit_log"

    audit_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    seq: Mapped[int] = mapped_column(BigInteger, Identity(), nullable=False, unique=True)
    patient_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False, index=True)
    input_refs: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    output_refs: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)


class Outcome(Base):
    """Outer-loop outcome capture (docs/11 §3). Append-only; the linked outputs +
    version snapshot let a later human-gated retraining run join outcomes to the
    exact artefacts that preceded them. Mirrors `schemas.outcome.Outcome`.
    """

    __tablename__ = "outcome"

    outcome_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    outcome_type: Mapped[str] = mapped_column(String, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    detail: Mapped[str] = mapped_column(String, nullable=False)
    code: Mapped[str | None] = mapped_column(String)
    linked_output_ids: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    source: Mapped[str] = mapped_column(String, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class VersionRegistryRow(Base):
    """Active model/ruleset/prompt/baseline-engine/schema versions (docs/06 §8).
    Changed only by human-gated releases; one active row per kind.
    """

    __tablename__ = "version_registry"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=NOW
    )
    created_by: Mapped[str] = mapped_column(String, nullable=False)

    __table_args__ = (
        Index(
            "uq_version_registry_active_kind",
            "kind",
            unique=True,
            postgresql_where=text("active"),
        ),
    )
