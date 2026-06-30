"""Consent gate — the T0.2 DoD test: 'consent gate blocks uncovered access'.

Covers the deny-by-default semantics (core/README.md decision 3): missing,
revoked, or non-covering consent all block — for every actor including system.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from core.auth import (
    Action,
    ConsentError,
    Operation,
    Principal,
    Role,
    authorize,
    consent_covers,
    granted_scopes,
    require_consent,
    require_operation_consent,
)
from schemas.consent import Consent, ConsentScope

GRANTED = datetime(2026, 1, 1, tzinfo=UTC)
REVOKED = datetime(2026, 2, 1, tzinfo=UTC)


def mk(scopes: list[ConsentScope], *, revoked: bool = False) -> Consent:
    return Consent(
        scope=scopes,
        version="consent-v1",
        granted_at=GRANTED,
        revoked_at=REVOKED if revoked else None,
    )


def test_covered_scope_passes() -> None:
    require_consent(mk([ConsentScope.VITALS]), ConsentScope.VITALS)  # no raise
    assert consent_covers(mk([ConsentScope.VITALS]), ConsentScope.VITALS)


def test_uncovered_scope_blocks() -> None:
    with pytest.raises(ConsentError):
        require_consent(mk([ConsentScope.DOCUMENTS]), ConsentScope.VITALS)


def test_revoked_consent_blocks_even_if_scope_listed() -> None:
    with pytest.raises(ConsentError):
        require_consent(mk([ConsentScope.VITALS], revoked=True), ConsentScope.VITALS)


def test_missing_consent_blocks() -> None:
    # Deny-by-default: no record on file is a denial, not a pass-through.
    with pytest.raises(ConsentError):
        require_consent(None, ConsentScope.VITALS)


def test_operation_to_scope_mapping() -> None:
    require_operation_consent(mk([ConsentScope.COPILOT]), Operation.COPILOT_QUERY)
    with pytest.raises(ConsentError):
        require_operation_consent(mk([ConsentScope.VITALS]), Operation.COPILOT_QUERY)


def test_granted_scopes_empty_when_revoked_or_absent() -> None:
    assert granted_scopes(mk([ConsentScope.VITALS, ConsentScope.COPILOT])) == {
        ConsentScope.VITALS,
        ConsentScope.COPILOT,
    }
    assert granted_scopes(None) == set()
    assert granted_scopes(mk([ConsentScope.VITALS], revoked=True)) == set()


def test_system_actor_is_not_consent_exempt() -> None:
    """RBAC lets the system pipeline ingest, but the consent gate still applies."""
    system = Principal(subject="ingestion-svc", roles=frozenset({Role.SYSTEM}))
    with pytest.raises(ConsentError):
        authorize(
            system,
            Action.INGEST_DATA,
            consent=mk([ConsentScope.DOCUMENTS]),
            required_scope=ConsentScope.VITALS,
        )
    # With covering consent the same call is allowed.
    authorize(
        system,
        Action.INGEST_DATA,
        consent=mk([ConsentScope.VITALS]),
        required_scope=ConsentScope.VITALS,
    )


def test_consent_error_carries_scope_and_patient() -> None:
    pid = uuid4()
    try:
        require_consent(None, ConsentScope.FORECAST, patient_id=pid)
    except ConsentError as exc:
        assert exc.required_scope == ConsentScope.FORECAST
        assert exc.patient_id == pid
        assert exc.http_status == 403
    else:  # pragma: no cover
        pytest.fail("expected ConsentError")
