"""T0.2 DoD: 'migrations apply'. Applies the baseline to a scratch DB, asserts the
expected tables exist, then reverses it cleanly back to empty.
"""

from __future__ import annotations

import os

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

_MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations")

EXPECTED_TABLES = {
    "patient_profile",
    "consent",
    "audit_log",
    "version_registry",
    "reading_node",
    "baseline_node",
    "deviation_node",
    "event_node",
    "condition_node",
    "medication_node",
    "allergy_node",
    "observation_node",
    "forecast_node",
    "output",
}


def _alembic_config() -> Config:
    # env.py injects the URL from DATABASE_URL (set by the scratch_db_url fixture),
    # so we only need to point alembic at the migration scripts.
    cfg = Config()
    cfg.set_main_option("script_location", _MIGRATIONS_DIR)
    return cfg


def test_baseline_applies_and_reverses(scratch_db_url: str) -> None:
    cfg = _alembic_config()

    command.upgrade(cfg, "head")
    engine = create_engine(scratch_db_url)
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert EXPECTED_TABLES <= tables, f"missing: {EXPECTED_TABLES - tables}"
    assert "alembic_version" in tables

    command.downgrade(cfg, "base")
    engine = create_engine(scratch_db_url)
    remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
    engine.dispose()
    assert remaining == set(), f"downgrade left tables behind: {remaining}"
