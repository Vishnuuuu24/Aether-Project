"""api-gateway — the single exposed edge (docs/07 §1, §9; docs/15 T7.1).

Every `/v1/*` request passes one deterministic pipeline before it reaches a backend:

    trace-id → authenticate (JWT) → resolve route → RBAC → ownership → forward

Guarantees:
  * No `/v1` route is reachable without a valid JWT (401 otherwise).
  * Access is RBAC-gated by role and ownership-gated by patient (403 otherwise);
    the per-scope consent record is still enforced downstream by each service.
  * Every response — success or error — carries `X-Trace-Id`.
  * Errors are RFC-7807 `application/problem+json` (docs/07 §9).

The gateway holds no business logic and returns no raw LLM output; it forwards to the
service that owns each route (`gateway.ROUTE_TABLE`). Health probes are unauthenticated.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from uuid import uuid4

import asyncpg
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from core.auth.errors import AuthError, ForbiddenError
from core.auth.jwt import JWTVerifier
from core.auth.principal import Principal
from core.auth.rbac import require_action
from core.observability import install_observability

from .gateway import Forwarder, HttpxForwarder, check_ownership, resolve

app = FastAPI(title="patient-copilot-api-gateway", version="0.0.1")

# Injectable seams (tests override these on app.state). The verifier is built lazily
# so importing the app needs no key material in the environment.
app.state.forwarder = HttpxForwarder()
app.state.jwt_verifier = None

_TRACE_HEADER = "X-Trace-Id"
# Hop-by-hop / edge-terminated headers never forwarded upstream.
_STRIP_HEADERS = {"host", "content-length", "authorization", "connection"}


def _verifier(request: Request) -> JWTVerifier:
    verifier: JWTVerifier | None = request.app.state.jwt_verifier
    if verifier is None:
        verifier = JWTVerifier.from_env()  # raises AuthError if no key configured
        request.app.state.jwt_verifier = verifier
    return verifier


def _authenticate(request: Request) -> Principal:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("missing or malformed bearer token")
    return _verifier(request).verify(token.strip())


def _problem(
    status_code: int, title: str, detail: str, *, instance: str, trace_id: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        content={
            "type": "about:blank",
            "title": title,
            "status": status_code,
            "detail": detail,
            "instance": instance,
            "trace": trace_id,
        },
        headers={_TRACE_HEADER: trace_id},
    )


def _forward_headers(request: Request, principal: Principal, trace_id: str) -> dict[str, str]:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _STRIP_HEADERS}
    headers[_TRACE_HEADER] = trace_id
    # The gateway terminates auth; it hands the verified identity to the trusted
    # backend (never the raw token). Downstream still applies its own consent gate.
    headers["X-Principal-Sub"] = principal.subject
    headers["X-Principal-Roles"] = " ".join(sorted(r.value for r in principal.roles))
    return headers


@app.middleware("http")
async def gateway_pipeline(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    trace_id = request.headers.get(_TRACE_HEADER) or uuid4().hex
    path = request.url.path

    # Health/ops probes are unauthenticated; still stamped with a trace id.
    if not path.startswith("/v1"):
        response = await call_next(request)
        response.headers[_TRACE_HEADER] = trace_id
        return response

    try:
        principal = _authenticate(request)
    except AuthError as exc:
        return _problem(401, "Unauthorized", str(exc), instance=path, trace_id=trace_id)

    rule = resolve(request.method, path)
    if rule is None:
        return _problem(
            404,
            "Not Found",
            f"no route for {request.method} {path}",
            instance=path,
            trace_id=trace_id,
        )

    try:
        if rule.action is not None:
            require_action(principal, rule.action)
        check_ownership(rule, path, principal)
    except ForbiddenError as exc:
        return _problem(403, "Forbidden", str(exc), instance=path, trace_id=trace_id)

    upstream_base = os.environ.get(rule.upstream_env)
    if not upstream_base:
        return _problem(
            502,
            "Bad Gateway",
            f"upstream {rule.upstream_env} not configured",
            instance=path,
            trace_id=trace_id,
        )

    body = await request.body()
    forwarder: Forwarder = request.app.state.forwarder
    try:
        forwarded = await forwarder.forward(
            method=request.method,
            upstream_base=upstream_base,
            path=path,
            headers=_forward_headers(request, principal, trace_id),
            body=body,
        )
    except Exception as exc:  # upstream unreachable / transport error
        return _problem(
            502, "Bad Gateway", f"upstream request failed: {exc}", instance=path, trace_id=trace_id
        )

    response = Response(
        content=forwarded.content,
        status_code=forwarded.status_code,
        media_type=forwarded.media_type,
    )
    response.headers[_TRACE_HEADER] = trace_id
    return response


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz(response: Response) -> dict[str, str | bool]:
    """Checks that Postgres and Qdrant are reachable (docker healthchecks + `make up`)."""
    checks: dict[str, bool] = {}

    db_url = os.environ.get("DATABASE_URL", "")
    try:
        conn = await asyncpg.connect(db_url, timeout=3)
        await conn.execute("SELECT 1")
        await conn.close()
        checks["postgres"] = True
    except Exception:
        checks["postgres"] = False

    try:
        qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
        host = qdrant_url.split("://")[-1].split(":")[0]
        port = int(qdrant_url.split(":")[-1])
        client = QdrantClient(host=host, port=port, timeout=3)
        client.get_collections()
        checks["qdrant"] = True
    except Exception:
        checks["qdrant"] = False

    all_ok = all(checks.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": all_ok, **checks}


# Registered LAST so it is the OUTERMOST middleware — it must observe (and trace) even
# the requests the auth pipeline rejects before they reach a backend.
install_observability(app, service="api-gateway")
