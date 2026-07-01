"""Shared model building blocks.

`NOW` is the `now()` server-side default reused across tables; `VersionedMixin`
carries the columns every versioned PSG node shares (docs/04 §3).
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid

NOW = text("now()")


class VersionedMixin:
    """Common columns for every versioned PSG node (docs/04 §3).

    Rows are append-only: a change writes a NEW version row with `supersedes`
    set; "current PSG" is the set of rows not referenced by any `supersedes`.
    """

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    patient_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    supersedes: Mapped[UUID | None] = mapped_column(Uuid)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=NOW
    )
    created_by: Mapped[str] = mapped_column(String, nullable=False)
