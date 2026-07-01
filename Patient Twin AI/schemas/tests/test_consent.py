"""Consent shape + covers() logic + round-trip (docs/04 §1)."""

from __future__ import annotations

from datetime import UTC, datetime

from schemas import Consent, ConsentScope


def mk(scopes: list[ConsentScope], *, revoked: bool = False) -> Consent:
    return Consent(
        scope=scopes,
        version="v1",
        granted_at=datetime(2026, 1, 1, tzinfo=UTC),
        revoked_at=datetime(2026, 2, 1, tzinfo=UTC) if revoked else None,
    )


def test_covers_true_for_granted_scope() -> None:
    assert mk([ConsentScope.VITALS, ConsentScope.COPILOT]).covers(ConsentScope.COPILOT)


def test_covers_false_for_ungranted_scope() -> None:
    assert not mk([ConsentScope.VITALS]).covers(ConsentScope.FORECAST)


def test_covers_false_when_revoked() -> None:
    assert not mk([ConsentScope.VITALS], revoked=True).covers(ConsentScope.VITALS)


def test_roundtrip_serialise_validate() -> None:
    c = mk([ConsentScope.VITALS, ConsentScope.DOCUMENTS])
    assert Consent.model_validate_json(c.model_dump_json()) == c
