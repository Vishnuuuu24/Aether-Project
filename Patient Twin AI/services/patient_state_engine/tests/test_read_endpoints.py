"""Twin/state read endpoints (docs/07 §4; docs/15 T6.2).

Every endpoint returns a consent-scoped slice of the projection (no raw signals),
gated per-resource: VITALS for baselines/deviations/events, FORECAST for forecast,
DOCUMENTS for observations/documents. Unknown patient → 404; missing the resource's
scope → 403 (deny-by-default), even if the patient has *other* scopes.
"""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient
from httpx import Response

from services.patient_state_engine.app.main import app, get_engine
from services.patient_state_engine.service import PatientStateEngine

from ._factories import (
    OCCURRED_AT,
    baseline,
    coded_result,
    deviation,
    documents_consent,
    event_candidate,
    forecast,
    forecast_consent,
    vitals_consent,
    wired_engine,
)


def _get(engine: PatientStateEngine, path: str) -> Response:
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        return TestClient(app).get(path)
    finally:
        app.dependency_overrides.clear()


def test_baselines_scoped_and_filtered() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent())
    engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)

    resp = _get(engine, f"/v1/patients/{pid}/baselines")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert "raw_ref" not in resp.text  # no reading-level data leaks

    assert len(_get(engine, f"/v1/patients/{pid}/baselines?metric=heart_rate").json()) == 1
    assert _get(engine, f"/v1/patients/{pid}/baselines?metric=steps").json() == []
    assert len(_get(engine, f"/v1/patients/{pid}/baselines?context=resting").json()) == 1
    assert _get(engine, f"/v1/patients/{pid}/baselines?context=active").json() == []


def test_deviations_since_filter() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent())
    engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)

    base = f"/v1/patients/{pid}/deviations"
    assert len(_get(engine, base).json()) == 1
    assert _get(engine, f"{base}?since=2026-07-01T00:00:00%2B00:00").json() == []  # after
    assert len(_get(engine, f"{base}?since=2026-01-01T00:00:00%2B00:00").json()) == 1  # before


def test_events_status_filter() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent())
    engine.commit_event(event_candidate(pid))

    assert len(_get(engine, f"/v1/patients/{pid}/events").json()) == 1
    assert len(_get(engine, f"/v1/patients/{pid}/events?status=active").json()) == 1
    # Only active events are surfaced; any other lifecycle status yields nothing.
    assert _get(engine, f"/v1/patients/{pid}/events?status=resolved").json() == []


def test_forecast_scoped_and_filtered() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=forecast_consent())
    engine.commit_forecast(forecast(pid))

    assert len(_get(engine, f"/v1/patients/{pid}/forecast").json()) == 1
    assert len(_get(engine, f"/v1/patients/{pid}/forecast?metric=heart_rate").json()) == 1
    assert len(_get(engine, f"/v1/patients/{pid}/forecast?horizon=3").json()) == 1
    assert _get(engine, f"/v1/patients/{pid}/forecast?horizon=5").json() == []


def test_observations_and_documents() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=documents_consent())
    engine.commit_coding(coded_result(pid))

    obs = _get(engine, f"/v1/patients/{pid}/observations")
    assert obs.status_code == 200
    assert len(obs.json()) == 1
    assert len(_get(engine, f"/v1/patients/{pid}/observations?code=4548-4").json()) == 1
    assert _get(engine, f"/v1/patients/{pid}/observations?code=0000-0").json() == []

    docs = _get(engine, f"/v1/patients/{pid}/documents")
    assert docs.status_code == 200
    body = docs.json()
    assert len(body) == 1
    assert body[0]["doc_type"] == "discharge_summary"
    assert body[0]["codes"]  # the coder's emitted codes are surfaced


def test_missing_resource_scope_is_403_even_with_other_scopes() -> None:
    pid = uuid4()
    # VITALS only: forecast (needs FORECAST) and documents (needs DOCUMENTS) are forbidden.
    engine, _, _ = wired_engine(pid, consent=vitals_consent())
    assert _get(engine, f"/v1/patients/{pid}/forecast").status_code == 403
    assert _get(engine, f"/v1/patients/{pid}/observations").status_code == 403
    assert _get(engine, f"/v1/patients/{pid}/documents").status_code == 403
    assert _get(engine, f"/v1/patients/{pid}/baselines").status_code == 200


def test_no_consent_at_all_is_403() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=None)
    assert _get(engine, f"/v1/patients/{pid}/baselines").status_code == 403


def test_unknown_patient_is_404() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent(), seed_profile=False)
    resp = _get(engine, f"/v1/patients/{pid}/baselines")
    assert resp.status_code == 404
