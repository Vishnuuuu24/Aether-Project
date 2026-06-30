"""Engine / session helpers. DATABASE_URL drives everything; on the host (alembic,
tests) it defaults to the docker-compose Postgres on localhost.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

# psycopg (v3) is the driver; normalise bare postgresql:// URLs to it.
_PSYCOPG = "postgresql+psycopg://"


def database_url(env: Mapping[str, str] | None = None) -> str:
    env = env if env is not None else os.environ
    url = env.get("DATABASE_URL")
    if url:
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", _PSYCOPG, 1)
        return url
    # Host default: the compose Postgres is published on localhost:5432.
    password = env.get("PG_PASSWORD", "")
    return f"{_PSYCOPG}hasa:{password}@localhost:5432/hasa"


def make_engine(url: str | None = None) -> Engine:
    return create_engine(url or database_url(), future=True, pool_pre_ping=True)


def make_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or make_engine(), class_=Session, expire_on_commit=False)
