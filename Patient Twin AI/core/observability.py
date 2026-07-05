"""Prometheus metrics + structured request logging (docs/07 §8; docs/15 T7.4).

`install_observability(app, service=...)` gives any service the same edge:
  * `GET /metrics` — Prometheus exposition (request count + latency).
  * a structured JSON access log per request carrying **only** method, normalised
    path, status, duration, and the trace id — never a body, query value, prompt,
    message, or reading value (Do: counts and IDs only; Don't: no PHI).
  * `X-Trace-Id` on every response (propagated from inbound or generated).

It is a **pure ASGI** middleware (not `BaseHTTPMiddleware`) so it does not buffer
streaming responses — the copilot SSE endpoint keeps streaming.

Path normalisation collapses UUID / numeric segments to `{id}` so patient ids never
reach a log line or a metric label (which would also explode cardinality).
"""

from __future__ import annotations

import json
import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.datastructures import MutableHeaders
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_NUM_RE = re.compile(r"^\d+$")
_TRACE_HEADER = b"x-trace-id"

REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["service", "method", "path", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["service", "method", "path"],
)

access_logger = logging.getLogger("patient_copilot.access")


def normalize_path(path: str) -> str:
    """Collapse id-like segments (UUIDs, numbers) to `{id}` — keeps metric cardinality
    bounded and keeps patient ids out of logs and labels.
    """
    out = []
    for segment in path.split("/"):
        out.append("{id}" if _UUID_RE.match(segment) or _NUM_RE.match(segment) else segment)
    return "/".join(out)


def _inbound_trace(scope: Scope) -> str | None:
    for key, value in scope.get("headers", []):
        if key == _TRACE_HEADER:
            return str(value.decode())
    return None


class ObservabilityMiddleware:
    def __init__(self, app: ASGIApp, *, service: str) -> None:
        self._app = app
        self._service = service

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        method: str = scope["method"]
        path = normalize_path(scope.get("path", ""))
        state: dict[str, object] = {"status": 500, "trace": _inbound_trace(scope) or uuid4().hex}
        start = perf_counter()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                headers = MutableHeaders(scope=message)
                existing = headers.get("x-trace-id")
                if existing:  # a downstream handler already set one (e.g. the gateway)
                    state["trace"] = existing
                else:
                    headers["x-trace-id"] = str(state["trace"])
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        finally:
            duration = perf_counter() - start
            status = str(state["status"])
            REQUESTS.labels(self._service, method, path, status).inc()
            LATENCY.labels(self._service, method, path).observe(duration)
            access_logger.info(
                json.dumps(
                    {
                        "service": self._service,
                        "method": method,
                        "path": path,  # normalised — no ids, no body, no query values
                        "status": state["status"],
                        "duration_ms": round(duration * 1000, 2),
                        "trace_id": state["trace"],
                    }
                )
            )


def install_observability(app: FastAPI, *, service: str) -> None:
    app.add_middleware(ObservabilityMiddleware, service=service)

    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.add_api_route("/metrics", metrics, methods=["GET"], include_in_schema=False)
