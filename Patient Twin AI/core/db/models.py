"""SQLAlchemy realisation of the contracts in `schemas/` (docs/04 §3).

These tables ARE the relational Patient State Graph plus the governance tables
(consent, audit, version registry). The pydantic models in `schemas/` remain the
single source of the *shape*; these ORM models are the storage projection of that
shape and must not diverge from it.

Immutability (docs/04 §3): PSG node rows are never updated in place. A change
writes a NEW version row with `supersedes` set; "current PSG" is the set of rows
not referenced by any `supersedes`. Enums are stored as strings (validated at the
pydantic layer) to keep migrations cheap.
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
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from .base import Base

_NOW = text("now()")


# ---------------------------------------------------------------------------
# Governance tables
# ---------------------------------------------------------------------------


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
        DateTime(timezone=True), nullable=False, server_default=_NOW
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
        DateTime(timezone=True), nullable=False, server_default=_NOW
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
        DateTime(timezone=True), nullable=False, server_default=_NOW
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


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------


class ReadingNode(Base):
    __tablename__ = "reading_node"

    reading_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    metric_code: Mapped[str] = mapped_column(String, nullable=False)
    # value is a scalar or structured JSON (e.g. sleep stages), so JSONB.
    value: Mapped[dict[str, Any] | float] = mapped_column(JSONB, nullable=False)
    unit: Mapped[str] = mapped_column(String, nullable=False)
    ts_tz: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_device: Mapped[str] = mapped_column(String, nullable=False)
    sqi: Mapped[float] = mapped_column(Float, nullable=False)
    context: Mapped[str] = mapped_column(String, nullable=False)
    included_in_baseline: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    ingest_adapter: Mapped[str] = mapped_column(String, nullable=False)
    raw_ref: Mapped[str | None] = mapped_column(String)


# ---------------------------------------------------------------------------
# Versioned PSG nodes
# ---------------------------------------------------------------------------


class VersionedMixin:
    """Common columns for every versioned PSG node (docs/04 §3)."""

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    supersedes: Mapped[UUID | None] = mapped_column(Uuid)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=_NOW
    )
    created_by: Mapped[str] = mapped_column(String, nullable=False)


class BaselineNode(VersionedMixin, Base):
    __tablename__ = "baseline_node"

    metric_code: Mapped[str] = mapped_column(String, nullable=False)
    context: Mapped[str] = mapped_column(String, nullable=False)
    method: Mapped[str] = mapped_column(String, nullable=False)
    center: Mapped[float] = mapped_column(Float, nullable=False)
    dispersion: Mapped[float] = mapped_column(Float, nullable=False)
    sample_n: Mapped[int] = mapped_column(Integer, nullable=False)
    window_spec: Mapped[str] = mapped_column(String, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_population_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False)


class DeviationNode(VersionedMixin, Base):
    __tablename__ = "deviation_node"

    metric_code: Mapped[str] = mapped_column(String, nullable=False)
    baseline_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    magnitude: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    z_robust: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_population_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False)


class EventNode(VersionedMixin, Base):
    __tablename__ = "event_node"

    type: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    onset_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    contributing_deviation_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(Uuid), nullable=False, default=list
    )


class ConditionNode(VersionedMixin, Base):
    __tablename__ = "condition_node"

    snomed_code: Mapped[str] = mapped_column(String, nullable=False)
    display: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    onset: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_document_id: Mapped[UUID | None] = mapped_column(Uuid)


class MedicationNode(VersionedMixin, Base):
    __tablename__ = "medication_node"

    rxnorm_code: Mapped[str] = mapped_column(String, nullable=False)
    display: Mapped[str] = mapped_column(String, nullable=False)
    dose: Mapped[str | None] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False)
    source_document_id: Mapped[UUID | None] = mapped_column(Uuid)


class AllergyNode(VersionedMixin, Base):
    __tablename__ = "allergy_node"

    substance_code: Mapped[str] = mapped_column(String, nullable=False)
    reaction: Mapped[str] = mapped_column(String, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)


class ObservationNode(VersionedMixin, Base):
    __tablename__ = "observation_node"

    loinc_code: Mapped[str] = mapped_column(String, nullable=False)
    display: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[str] = mapped_column(String, nullable=False)
    unit: Mapped[str] = mapped_column(String, nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_document_id: Mapped[UUID | None] = mapped_column(Uuid)


class ForecastNode(VersionedMixin, Base):
    __tablename__ = "forecast_node"

    metric_code: Mapped[str] = mapped_column(String, nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    points: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    intervals: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    method: Mapped[str] = mapped_column(String, nullable=False)
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# ---------------------------------------------------------------------------
# Persisted outputs (docs/04 §6 — every output, incl. abstain/suppress, stored)
# ---------------------------------------------------------------------------


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
