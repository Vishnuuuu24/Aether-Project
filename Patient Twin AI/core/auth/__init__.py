"""core.auth — JWT verification, RBAC, and consent enforcement (docs/06, docs/07).

Two independent gates that BOTH must pass before a data-processing operation runs:
  1. RBAC      — is this caller allowed to attempt the action?  (`require_action`)
  2. Consent   — has the patient consented to this processing?  (`require_consent`)

`authorize()` runs them in that order, deny-by-default.
"""

from __future__ import annotations

from uuid import UUID

from schemas.consent import Consent, ConsentScope

from .consent_gate import (
    OPERATION_SCOPE,
    Operation,
    consent_covers,
    granted_scopes,
    require_consent,
    require_operation_consent,
)
from .errors import AuthError, ConsentError, ForbiddenError
from .jwt import JWTVerifier
from .principal import Principal, Role
from .rbac import ROLE_PERMISSIONS, Action, can, require_action


def authorize(
    principal: Principal,
    action: Action,
    *,
    consent: Consent | None = None,
    required_scope: ConsentScope | None = None,
    patient_id: UUID | None = None,
) -> None:
    """Full gate. RBAC first, then consent (if the action processes patient data).

    Raises ForbiddenError on an RBAC denial or ConsentError on a consent denial.
    """
    require_action(principal, action)
    if required_scope is not None:
        require_consent(consent, required_scope, patient_id=patient_id)


__all__ = [
    "AuthError",
    "ForbiddenError",
    "ConsentError",
    "JWTVerifier",
    "Principal",
    "Role",
    "Action",
    "ROLE_PERMISSIONS",
    "can",
    "require_action",
    "Operation",
    "OPERATION_SCOPE",
    "consent_covers",
    "require_consent",
    "require_operation_consent",
    "granted_scopes",
    "authorize",
]
