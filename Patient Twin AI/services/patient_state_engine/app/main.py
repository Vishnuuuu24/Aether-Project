"""patient-state-engine API (docs/07; T1.4).

`GET /v1/state/{patient_id}` — the consent-scoped PSG projection. It carries no raw
signals and no reading-level data (CLAUDE.md principle 2). Deny-by-default: a patient
with no consent scope in force gets 403; an unknown patient gets 404.

Persistence is config-switched, not forked (CLAUDE.md): `PERSISTENCE_BACKEND=memory`
(default, dev) uses in-memory stores; `postgres` opens a per-request transactional
session and binds the SAME engine to `SqlAlchemyPSGStore` + `SqlAlchemyAuditStore`
and DB-backed consent/profile providers (docs/15 T7.2).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session, sessionmaker

from core.audit import AuditWriter, InMemoryAuditStore
from core.auth.errors import ConsentError
from core.observability import install_observability
from core.versioning import VersionRegistry
from schemas.psg import (
    BaselineSummary,
    DeviationSummary,
    DocumentSummary,
    EventSummary,
    ForecastSummary,
    ObservationSummary,
    PSGProjection,
)
from schemas.reading import MeasurementContext, MetricCode

from ..consent import StaticConsentProvider
from ..profile import StaticProfileProvider
from ..service import PatientStateEngine, ProfileNotFoundError
from ..store import InMemoryPSGStore
from ..wiring import build_sql_engine

app = FastAPI(title="patient-copilot-state-engine", version="0.0.1")
install_observability(app, service="state-engine")

_versions = VersionRegistry.from_env().current()
_persistence = os.environ.get("PERSISTENCE_BACKEND", "memory").lower()

# In-memory dev wiring (default). Overridden per-request when PERSISTENCE_BACKEND=postgres.
_consent_provider = StaticConsentProvider()
_profile_provider = StaticProfileProvider()
_memory_engine = PatientStateEngine(
    store=InMemoryPSGStore(),
    consent_provider=_consent_provider,
    audit_writer=AuditWriter(InMemoryAuditStore()),
    versions=_versions,
    profile_provider=_profile_provider,
)

_session_factory: sessionmaker[Session] | None = None


def _get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        from core.db import make_session_factory

        _session_factory = make_session_factory()
    return _session_factory


def get_engine() -> Iterator[PatientStateEngine]:
    """Transaction-per-request when DB-backed: commit on success, roll back on any
    error, always close. The in-memory path yields the shared dev engine unchanged.
    """
    if _persistence == "postgres":
        session = _get_session_factory()()
        try:
            yield build_sql_engine(session, _versions)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    else:
        yield _memory_engine


_Engine = Annotated[PatientStateEngine, Depends(get_engine)]


@contextmanager
def _read_errors() -> Iterator[None]:
    """Uniform 404/403 mapping for every scoped read (mirrors `/state`):
    unknown patient → 404 (never disclosed as merely 'forbidden'); missing the
    resource's consent scope → 403 (deny-by-default).
    """
    try:
        yield
    except ProfileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient not found"
        ) from exc
    except ConsentError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz() -> dict[str, bool]:
    return {"ready": True}


@app.get("/v1/state/{patient_id}")
async def get_state(patient_id: UUID, engine: _Engine) -> PSGProjection:
    with _read_errors():
        return engine.build_projection(patient_id)


@app.get("/v1/patients/{patient_id}/baselines")
async def get_baselines(
    patient_id: UUID,
    engine: _Engine,
    metric: MetricCode | None = None,
    context: MeasurementContext | None = None,
) -> list[BaselineSummary]:
    with _read_errors():
        return engine.read_baselines(patient_id, metric=metric, context=context)


@app.get("/v1/patients/{patient_id}/deviations")
async def get_deviations(
    patient_id: UUID, engine: _Engine, since: datetime | None = None
) -> list[DeviationSummary]:
    with _read_errors():
        return engine.read_deviations(patient_id, since=since)


@app.get("/v1/patients/{patient_id}/events")
async def get_events(
    patient_id: UUID, engine: _Engine, status: str | None = None
) -> list[EventSummary]:
    with _read_errors():
        return engine.read_events(patient_id, status=status)


@app.get("/v1/patients/{patient_id}/forecast")
async def get_forecast(
    patient_id: UUID,
    engine: _Engine,
    metric: MetricCode | None = None,
    horizon: int | None = None,
) -> list[ForecastSummary]:
    with _read_errors():
        return engine.read_forecasts(patient_id, metric=metric, horizon=horizon)


@app.get("/v1/patients/{patient_id}/observations")
async def get_observations(
    patient_id: UUID, engine: _Engine, code: str | None = None
) -> list[ObservationSummary]:
    with _read_errors():
        return engine.read_observations(patient_id, code=code)


@app.get("/v1/patients/{patient_id}/documents")
async def get_documents(patient_id: UUID, engine: _Engine) -> list[DocumentSummary]:
    with _read_errors():
        return engine.read_documents(patient_id)
