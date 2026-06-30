"""Role-based access control (docs/07 §1).

RBAC answers "is this *caller* allowed to attempt this action?". It is the first
gate; the consent gate (`core.auth.consent_gate`) is the second and independent.

The permission matrix below is a config stub — the action surface firms up as the
API in docs/07 lands (Sprint 4). It is deliberately deny-by-default: an action not
listed for a role is forbidden. This is platform policy, NOT clinical content, so
it is safe to define here (contrast: red-flag rules / SQI thresholds, which stay
stubs per CLAUDE.md).
"""

from __future__ import annotations

from enum import Enum

from .errors import ForbiddenError
from .principal import Principal, Role


class Action(str, Enum):
    INGEST_DATA = "ingest_data"
    READ_STATE = "read_state"
    COPILOT_QUERY = "copilot_query"
    MANAGE_CONSENT = "manage_consent"
    READ_AUDIT = "read_audit"
    READ_ESCALATIONS = "read_escalations"
    ACK_ESCALATION = "ack_escalation"
    RECORD_OUTCOME = "record_outcome"
    MANAGE_VERSIONS = "manage_versions"


ROLE_PERMISSIONS: dict[Role, frozenset[Action]] = {
    Role.PATIENT: frozenset(
        {Action.INGEST_DATA, Action.READ_STATE, Action.COPILOT_QUERY, Action.MANAGE_CONSENT}
    ),
    Role.CLINICIAN: frozenset(
        {
            Action.READ_STATE,
            Action.READ_ESCALATIONS,
            Action.ACK_ESCALATION,
            Action.READ_AUDIT,
            Action.RECORD_OUTCOME,
        }
    ),
    Role.SYSTEM: frozenset(
        {Action.INGEST_DATA, Action.READ_STATE, Action.COPILOT_QUERY, Action.RECORD_OUTCOME}
    ),
    Role.ADMIN: frozenset(
        {Action.READ_AUDIT, Action.MANAGE_VERSIONS, Action.MANAGE_CONSENT, Action.READ_ESCALATIONS}
    ),
}


def can(principal: Principal, action: Action) -> bool:
    return any(action in ROLE_PERMISSIONS.get(role, frozenset()) for role in principal.roles)


def require_action(principal: Principal, action: Action) -> None:
    if not can(principal, action):
        roles = ", ".join(sorted(r.value for r in principal.roles)) or "<none>"
        raise ForbiddenError(f"RBAC: roles [{roles}] may not perform '{action.value}'")
