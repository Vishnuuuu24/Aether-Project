"""Typed auth/consent errors. Carry an HTTP status so the api-gateway can map
them to RFC 7807 problem+json (docs/07 §9) without re-deriving intent.
"""

from __future__ import annotations

from schemas.consent import ConsentScope


class AuthError(Exception):
    """Authentication failed: missing/invalid/expired token. → 401."""

    http_status = 401


class ForbiddenError(Exception):
    """Authenticated but not permitted (RBAC). → 403."""

    http_status = 403


class ConsentError(ForbiddenError):
    """The patient has not consented to the scope this operation requires. → 403.

    Distinct from a plain RBAC denial: the caller may be perfectly authorized,
    but the *patient's* consent does not cover the processing.
    """

    def __init__(
        self,
        message: str,
        *,
        required_scope: ConsentScope | None = None,
        patient_id: object | None = None,
    ) -> None:
        super().__init__(message)
        self.required_scope = required_scope
        self.patient_id = patient_id
