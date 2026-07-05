"""Observability edge: /metrics, structured logging, trace ids, no PHI, streaming-safe
(docs/07 §8; docs/15 T7.4).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from core.observability import install_observability, normalize_path


def _app() -> FastAPI:
    app = FastAPI()
    install_observability(app, service="test-svc")

    @app.post("/v1/patients/{pid}/echo")
    async def echo(pid: str, request: Request) -> dict[str, bool]:
        await request.body()  # consume the (PHI-bearing) body; it must never be logged
        return {"ok": True}

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def gen() -> AsyncIterator[bytes]:
            for i in range(3):
                yield f"chunk-{i}\n".encode()

        return StreamingResponse(gen(), media_type="text/plain")

    return app


def test_metrics_endpoint_exposes_prometheus() -> None:
    client = TestClient(_app())
    client.get("/stream")  # produce at least one observation
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "http_requests_total" in resp.text


def test_every_response_carries_a_trace_id() -> None:
    resp = TestClient(_app()).get("/stream")
    assert resp.headers["X-Trace-Id"]


def test_inbound_trace_id_is_propagated() -> None:
    resp = TestClient(_app()).get("/stream", headers={"X-Trace-Id": "trace-xyz"})
    assert resp.headers["X-Trace-Id"] == "trace-xyz"


def test_access_log_is_structured_normalized_and_phi_free(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="patient_copilot.access")
    secret = "patient-reported-symptom-XYZZY"
    pid = "11111111-1111-1111-1111-111111111111"

    TestClient(_app()).post(f"/v1/patients/{pid}/echo", json={"message": secret, "reading": 137})

    logs = [
        json.loads(r.getMessage()) for r in caplog.records if r.name == "patient_copilot.access"
    ]
    echoed = [entry for entry in logs if entry["path"] == "/v1/patients/{id}/echo"]
    assert echoed, "expected a structured access log with the patient id normalised to {id}"
    entry = echoed[0]
    assert entry["method"] == "POST"
    assert entry["status"] == 200
    assert entry["trace_id"]
    # No PHI reaches the log: neither the request body nor the patient id.
    assert secret not in caplog.text
    assert pid not in caplog.text


def test_streaming_is_not_buffered_by_middleware() -> None:
    # The pure-ASGI middleware must not consume a streaming body (copilot SSE relies on this).
    resp = TestClient(_app()).get("/stream")
    assert resp.status_code == 200
    assert resp.text == "chunk-0\nchunk-1\nchunk-2\n"


def test_normalize_path_collapses_id_segments() -> None:
    assert (
        normalize_path("/v1/patients/11111111-1111-1111-1111-111111111111/state")
        == "/v1/patients/{id}/state"
    )
    assert normalize_path("/v1/escalations/12345/ack") == "/v1/escalations/{id}/ack"
    assert normalize_path("/v1/audit") == "/v1/audit"  # non-id segments untouched


def test_real_service_app_exposes_metrics() -> None:
    from services.governance_service.app.main import app as governance_app

    resp = TestClient(governance_app).get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text
