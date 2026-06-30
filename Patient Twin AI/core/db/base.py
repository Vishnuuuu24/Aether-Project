"""Declarative base for all ORM models. One metadata object → one migration set."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
