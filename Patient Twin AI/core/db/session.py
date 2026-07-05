"""Engine / session helpers. DATABASE_URL drives everything; on the host (alembic,
tests) it defaults to the docker-compose Postgres on localhost.

Also the ONE place the memory↔postgres switch lives: `request_session` is a FastAPI
dependency every service reuses so a request's writes are one atomic transaction and
the switch is config (`PERSISTENCE_BACKEND`), not forked code (CLAUDE.md).
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

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


def persistence_backend(env: Mapping[str, str] | None = None) -> str:
    """`postgres` when DB-backed, else `memory` (dev default). Read at request time so
    the switch is runtime config, not import-time."""
    env = env if env is not None else os.environ
    return env.get("PERSISTENCE_BACKEND", "memory").strip().lower()


@contextmanager
def transactional_session(factory: sessionmaker[Session]) -> Iterator[Session]:
    """One request = one transaction: commit on success, roll back on any error,
    always close. Callers write through the store interfaces; the boundary is here."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


_request_factory: sessionmaker[Session] | None = None


def _cached_factory() -> sessionmaker[Session]:
    global _request_factory
    if _request_factory is None:
        _request_factory = make_session_factory()
    return _request_factory


def request_session() -> Iterator[Session | None]:
    """FastAPI dependency shared by every service. Yields a transactional `Session`
    in `postgres` mode (FastAPI caches it per request, so all of a service's writers
    share ONE transaction / one audit-chain append) or `None` in `memory` mode, where
    each service falls back to its in-memory dev store.
    """
    if persistence_backend() != "postgres":
        yield None
        return
    with transactional_session(_cached_factory()) as session:
        yield session
