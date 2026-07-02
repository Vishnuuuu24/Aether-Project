"""GET /v1/state/{patient_id} — scoped projection, 403 no-consent, 404 unknown."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from services.patient_state_engine.app.main import app, get_engine

from ._factories import OCCURRED_AT, baseline, deviation, vitals_consent, wired_engine


def test_get_state_returns_scoped_projection() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent())
    engine.commit_deviation(baseline(pid), deviation(pid), occurred_at=OCCURRED_AT)

    app.dependency_overrides[get_engine] = lambda: engine
    try:
        resp = TestClient(app).get(f"/v1/state/{pid}")
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["consent_scope"] == ["vitals"]
    assert len(body["baselines"]) == 1
    assert len(body["recent_deviations"]) == 1
    assert "raw_ref" not in resp.text


def test_get_state_403_without_consent() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=None)
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        resp = TestClient(app).get(f"/v1/state/{pid}")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 403


def test_get_state_404_unknown_patient() -> None:
    pid = uuid4()
    engine, _, _ = wired_engine(pid, consent=vitals_consent(), seed_profile=False)
    app.dependency_overrides[get_engine] = lambda: engine
    try:
        resp = TestClient(app).get(f"/v1/state/{pid}")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 404
