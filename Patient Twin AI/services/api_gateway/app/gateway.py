"""Gateway routing table + edge authorization (docs/07 §1, §4-7; docs/15 T7.1).

The api-gateway is the ONLY exposed service (docs/08). It authenticates every `/v1`
request, applies RBAC and patient-ownership scoping, then forwards to the backend
service that owns the route. This module holds the deterministic parts — the route
table, resolution, the ownership check, and the forwarder seam — so the middleware
in `main.py` stays a thin pipeline.

Two independent gates, both server-side (never trust client-supplied scopes):
  * RBAC — may this *caller's role* attempt this action at all (`core.auth.rbac`).
  * Ownership — a patient principal may only touch **their own** `patient_id`; a
    clinician/admin/system may cross patients (already RBAC-gated). The per-scope
    *consent* record is enforced a second time downstream by each service's own
    consent gate (deny-by-default) — the gateway never bypasses it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from core.auth.errors import ForbiddenError
from core.auth.principal import Principal, Role
from core.auth.rbac import Action

# Roles allowed to act across patients; a bare patient is confined to its own record.
_CROSS_PATIENT_ROLES = frozenset({Role.CLINICIAN, Role.ADMIN, Role.SYSTEM})


@dataclass(frozen=True)
class RouteRule:
    methods: frozenset[str]
    pattern: re.Pattern[str]
    action: Action | None  # None => any authenticated principal (e.g. read versions)
    upstream_env: str  # env var naming the upstream base URL
    patient_scoped: bool  # path carries a `pid` group the caller must own


def _rule(methods: str, pattern: str, action: Action | None, upstream_env: str) -> RouteRule:
    compiled = re.compile(pattern)
    patient_scoped = "(?P<pid>" in pattern
    return RouteRule(
        methods=frozenset(methods.split()),
        pattern=compiled,
        action=action,
        upstream_env=upstream_env,
        patient_scoped=patient_scoped,
    )


# First match wins. Every exposed `/v1` route in docs/07 is listed; anything not here
# is a 404 at the edge (deny-by-default surface).
ROUTE_TABLE: tuple[RouteRule, ...] = (
    _rule("POST", r"^/v1/ingest/documents$", Action.INGEST_DATA, "DOC_CODING_URL"),
    _rule("POST", r"^/v1/ingest/replay$", Action.INGEST_DATA, "INGESTION_URL"),
    _rule("POST", r"^/v1/ingest/adapters/[^/]+/webhook$", Action.INGEST_DATA, "INGESTION_URL"),
    _rule("POST", r"^/v1/ingest/readings$", Action.INGEST_DATA, "INGESTION_URL"),
    _rule(
        "POST",
        r"^/v1/patients/(?P<pid>[^/]+)/copilot/query(:stream)?$",
        Action.COPILOT_QUERY,
        "COPILOT_URL",
    ),
    _rule(
        "GET POST DELETE",
        r"^/v1/patients/(?P<pid>[^/]+)/consent$",
        Action.MANAGE_CONSENT,
        "GOVERNANCE_URL",
    ),
    _rule(
        "GET",
        r"^/v1/patients/(?P<pid>[^/]+)/"
        r"(state|baselines|deviations|events|forecast|observations|documents)$",
        Action.READ_STATE,
        "STATE_ENGINE_URL",
    ),
    _rule("GET", r"^/v1/escalations$", Action.READ_ESCALATIONS, "COPILOT_URL"),
    _rule("POST", r"^/v1/escalations/[^/]+/ack$", Action.ACK_ESCALATION, "COPILOT_URL"),
    _rule("GET", r"^/v1/audit$", Action.READ_AUDIT, "GOVERNANCE_URL"),
    _rule("POST", r"^/v1/outcomes$", Action.RECORD_OUTCOME, "GOVERNANCE_URL"),
    _rule("GET", r"^/v1/versions$", None, "GOVERNANCE_URL"),
)


def resolve(method: str, path: str) -> RouteRule | None:
    for rule in ROUTE_TABLE:
        if method in rule.methods and rule.pattern.match(path):
            return rule
    return None


def check_ownership(rule: RouteRule, path: str, principal: Principal) -> None:
    """A bare patient may only reach its own `patient_id`. Raises ForbiddenError
    otherwise. Cross-patient roles (clinician/admin/system) are already RBAC-gated.
    """
    if not rule.patient_scoped or principal.roles & _CROSS_PATIENT_ROLES:
        return
    match = rule.pattern.match(path)
    pid_raw = match.group("pid") if match else None
    if pid_raw is None:
        return
    try:
        target = UUID(pid_raw)
    except ValueError as exc:
        raise ForbiddenError("invalid patient id in path") from exc
    if principal.patient_id != target:
        raise ForbiddenError("a patient may only access their own record")


@dataclass(frozen=True)
class ForwardedResponse:
    status_code: int
    content: bytes
    media_type: str


class Forwarder(Protocol):
    async def forward(
        self,
        *,
        method: str,
        upstream_base: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> ForwardedResponse: ...


class HttpxForwarder:
    """Default forwarder: proxies the request to the resolved upstream over HTTP.
    httpx is imported lazily so importing the gateway needs no HTTP client.
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        self._timeout_s = timeout_s

    async def forward(
        self,
        *,
        method: str,
        upstream_base: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> ForwardedResponse:
        import httpx

        url = upstream_base.rstrip("/") + path
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.request(method, url, content=body, headers=dict(headers))
        return ForwardedResponse(
            status_code=resp.status_code,
            content=resp.content,
            media_type=resp.headers.get("content-type", "application/json"),
        )
