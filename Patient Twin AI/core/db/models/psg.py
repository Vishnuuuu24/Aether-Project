"""Versioned Patient State Graph nodes (docs/04 §3).

The personal-baseline state: baselines, deviations from them, multi-metric
events, the document-derived clinical nodes (conditions / medications /
allergies / observations), and forecasts. All inherit `VersionedMixin` and are
append-only.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from ..base import Base
from .common import VersionedMixin


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
