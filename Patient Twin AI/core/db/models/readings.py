"""The per-reading table — normalised vitals data points (docs/04 §2).

A reading is the raw-ish ingest record. `value` is JSONB so it holds either a
scalar or a structured value (e.g. sleep stages); `raw_ref` points at the raw
signal window in object storage and is NEVER projected downstream of features.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

from ..base import Base


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
