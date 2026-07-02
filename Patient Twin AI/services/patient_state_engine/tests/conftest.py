"""DB integration fixture: a throwaway scratch database per test, skipping cleanly
when Postgres is not reachable (mirrors core/db/tests/conftest.py).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError


def _base_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        password = os.environ.get("PG_PASSWORD", "dev-local-pg-pw")
        url = f"postgresql+psycopg://hasa:{password}@localhost:5432/hasa"
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@pytest.fixture
def scratch_db_url(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    base = make_url(_base_url())
    admin = create_engine(base, isolation_level="AUTOCOMMIT")
    name = f"t14_{uuid4().hex[:12]}"
    try:
        with admin.connect() as conn:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except OperationalError:
        admin.dispose()
        pytest.skip("Postgres not reachable — skipping DB integration test")

    url = base.set(database=name).render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", url)
    try:
        yield url
    finally:
        with admin.connect() as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": name},
            )
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()
