"""api-gateway edge pipeline (docs/07 §1, §9; docs/15 T7.1).

Auth (401), RBAC + patient ownership (403), unknown route (404), X-Trace-Id on every
response, RFC-7807 problem+json, and forwarding to the resolved upstream — with the
JWT verifier and the forwarder injected so no real backend or key material is needed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from uuid import uuid4

import jwt
import pytest
from fastapi.testclient import TestClient

from core.auth.jwt import JWTVerifier
from services.api_gateway.app.gateway import ForwardedResponse, HttpxForwarder
from services.api_gateway.app.main import app

_SECRET = "test-secret"
_UPSTREAMS = (
    "STATE_ENGINE_URL",
    "COPILOT_URL",
    "GOVERNANCE_URL",
    "INGESTION_URL",
    "DOC_CODING_URL",
)


class _FakeForwarder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def forward(
        self,
        *,
        method: str,
        upstream_base: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> ForwardedResponse:
        self.calls.append(
            {
                "method": method,
                "upstream_base": upstream_base,
                "path": path,
                "headers": dict(headers),
            }
        )
        return ForwardedResponse(
            status_code=200, content=b'{"ok":true}', media_type="application/json"
        )


def _token(roles: list[str], *, patient_id: object = None, sub: str = "user-1") -> str:
    claims: dict[str, object] = {"sub": sub, "exp": int(time.time()) + 300, "roles": roles}
    if patient_id is not None:
        claims["patient_id"] = str(patient_id)
    return jwt.encode(claims, _SECRET, algorithm="HS256")


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, _FakeForwarder]]:
    app.state.jwt_verifier = JWTVerifier(key=_SECRET, algorithms=("HS256",))
    forwarder = _FakeForwarder()
    app.state.forwarder = forwarder
    for var in _UPSTREAMS:
        monkeypatch.setenv(var, "http://upstream")
    try:
        yield TestClient(app), forwarder
    finally:
        app.state.jwt_verifier = None
        app.state.forwarder = HttpxForwarder()


def test_missing_token_is_401_problem_json(gateway: tuple[TestClient, _FakeForwarder]) -> None:
    client, forwarder = gateway
    resp = client.get(f"/v1/patients/{uuid4()}/state")
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert {"type", "title", "status", "detail", "instance", "trace"} <= body.keys()
    assert body["status"] == 401
    assert resp.headers["X-Trace-Id"]
    assert forwarder.calls == []  # never reached the backend


def test_invalid_token_is_401(gateway: tuple[TestClient, _FakeForwarder]) -> None:
    client, _ = gateway
    resp = client.get(f"/v1/patients/{uuid4()}/state", headers=_auth("not-a-jwt"))
    assert resp.status_code == 401


def test_patient_reads_own_state_and_is_forwarded(
    gateway: tuple[TestClient, _FakeForwarder],
) -> None:
    client, forwarder = gateway
    pid = uuid4()
    resp = client.get(
        f"/v1/patients/{pid}/state", headers=_auth(_token(["patient"], patient_id=pid))
    )
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert len(forwarder.calls) == 1
    call = forwarder.calls[0]
    assert call["upstream_base"] == "http://upstream"
    assert call["path"] == f"/v1/patients/{pid}/state"
    headers = call["headers"]
    assert isinstance(headers, dict)
    assert headers["X-Principal-Roles"] == "patient"  # verified identity forwarded
    assert "authorization" not in {k.lower() for k in headers}  # raw token not leaked upstream


def test_patient_cannot_read_another_patient_is_403(
    gateway: tuple[TestClient, _FakeForwarder],
) -> None:
    client, forwarder = gateway
    resp = client.get(
        f"/v1/patients/{uuid4()}/state",  # a different patient
        headers=_auth(_token(["patient"], patient_id=uuid4())),
    )
    assert resp.status_code == 403
    assert forwarder.calls == []


def test_clinician_may_cross_patients(gateway: tuple[TestClient, _FakeForwarder]) -> None:
    client, forwarder = gateway
    resp = client.get(f"/v1/patients/{uuid4()}/state", headers=_auth(_token(["clinician"])))
    assert resp.status_code == 200
    assert len(forwarder.calls) == 1


def test_rbac_forbids_action_outside_role(gateway: tuple[TestClient, _FakeForwarder]) -> None:
    client, forwarder = gateway
    # A patient may not read the audit trail (clinician/admin only).
    resp = client.get("/v1/audit", headers=_auth(_token(["patient"], patient_id=uuid4())))
    assert resp.status_code == 403
    assert forwarder.calls == []


def test_unknown_v1_route_is_404(gateway: tuple[TestClient, _FakeForwarder]) -> None:
    client, _ = gateway
    resp = client.get("/v1/not/a/route", headers=_auth(_token(["admin"])))
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")


def test_versions_allows_any_authenticated_principal(
    gateway: tuple[TestClient, _FakeForwarder],
) -> None:
    client, forwarder = gateway
    resp = client.get("/v1/versions", headers=_auth(_token(["patient"], patient_id=uuid4())))
    assert resp.status_code == 200
    assert len(forwarder.calls) == 1


def test_trace_id_is_propagated_from_inbound_header(
    gateway: tuple[TestClient, _FakeForwarder],
) -> None:
    client, _ = gateway
    resp = client.get(
        "/v1/versions",
        headers={**_auth(_token(["admin"])), "X-Trace-Id": "trace-abc"},
    )
    assert resp.headers["X-Trace-Id"] == "trace-abc"


def test_health_is_unauthenticated_and_traced(gateway: tuple[TestClient, _FakeForwarder]) -> None:
    client, _ = gateway
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.headers["X-Trace-Id"]


def test_upstream_not_configured_is_502(
    gateway: tuple[TestClient, _FakeForwarder], monkeypatch: pytest.MonkeyPatch
) -> None:
    client, forwarder = gateway
    monkeypatch.delenv("GOVERNANCE_URL", raising=False)
    resp = client.get("/v1/versions", headers=_auth(_token(["admin"])))
    assert resp.status_code == 502
    assert forwarder.calls == []
