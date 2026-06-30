"""RBAC — role→action gating, and that RBAC runs before the consent gate."""

from __future__ import annotations

import pytest

from core.auth import Action, ForbiddenError, Principal, Role, authorize, can, require_action

PATIENT = Principal(subject="p", roles=frozenset({Role.PATIENT}))
CLINICIAN = Principal(subject="c", roles=frozenset({Role.CLINICIAN}))
ADMIN = Principal(subject="a", roles=frozenset({Role.ADMIN}))


def test_patient_may_ingest_and_query() -> None:
    require_action(PATIENT, Action.INGEST_DATA)
    require_action(PATIENT, Action.COPILOT_QUERY)


def test_patient_may_not_read_audit() -> None:
    assert not can(PATIENT, Action.READ_AUDIT)
    with pytest.raises(ForbiddenError):
        require_action(PATIENT, Action.READ_AUDIT)


def test_clinician_reads_audit_admin_manages_versions() -> None:
    require_action(CLINICIAN, Action.READ_AUDIT)
    require_action(ADMIN, Action.MANAGE_VERSIONS)
    assert not can(CLINICIAN, Action.MANAGE_VERSIONS)


def test_no_roles_denied() -> None:
    nobody = Principal(subject="x", roles=frozenset())
    with pytest.raises(ForbiddenError):
        require_action(nobody, Action.READ_STATE)


def test_authorize_rbac_denial_precedes_consent() -> None:
    # RBAC fails first → ForbiddenError, never reaches the consent check.
    with pytest.raises(ForbiddenError):
        authorize(PATIENT, Action.MANAGE_VERSIONS)
