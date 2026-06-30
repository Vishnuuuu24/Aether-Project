"""Consent enforcement — the non-negotiable second gate (CLAUDE.md; docs/06).

"No processing without a valid, non-revoked consent covering the relevant scope."
This applies to EVERY actor, including the internal `system` pipeline: internal
processing is still processing. Deny-by-default — if there is no consent, or it is
revoked, or it does not cover the required scope, the operation is blocked.

The shape of consent lives in `schemas.consent`; this module owns the *decision*.
"""

from __future__ import annotations

from enum import Enum
from uuid import UUID

from schemas.consent import Consent, ConsentScope

from .errors import ConsentError


def consent_covers(consent: Consent | None, required: ConsentScope) -> bool:
    """Pure predicate. True iff a non-revoked consent grants `required`."""
    return consent is not None and consent.covers(required)


def require_consent(
    consent: Consent | None,
    required: ConsentScope,
    *,
    patient_id: UUID | None = None,
) -> None:
    """Raise ConsentError unless the patient consented to `required` and has not
    revoked. Deny-by-default: a missing consent record is a denial, not a pass.
    """
    if consent_covers(consent, required):
        return
    if consent is None:
        reason = "no consent record on file"
    elif consent.revoked_at is not None:
        reason = f"consent revoked at {consent.revoked_at.isoformat()}"
    else:
        reason = (
            f"scope '{required.value}' not granted (granted: {[s.value for s in consent.scope]})"
        )
    raise ConsentError(
        f"consent gate denied: {reason}", required_scope=required, patient_id=patient_id
    )


class Operation(str, Enum):
    """Data-processing operations and the consent scope each requires.

    Consent-management and audit/governance endpoints are intentionally absent:
    they are governed by RBAC, not by this gate (a patient must be able to revoke
    consent even after it is revoked).
    """

    INGEST_VITALS = "ingest_vitals"
    INGEST_DOCUMENT = "ingest_document"
    COPILOT_QUERY = "copilot_query"
    FORECAST_READ = "forecast_read"


OPERATION_SCOPE: dict[Operation, ConsentScope] = {
    Operation.INGEST_VITALS: ConsentScope.VITALS,
    Operation.INGEST_DOCUMENT: ConsentScope.DOCUMENTS,
    Operation.COPILOT_QUERY: ConsentScope.COPILOT,
    Operation.FORECAST_READ: ConsentScope.FORECAST,
}


def require_operation_consent(
    consent: Consent | None,
    operation: Operation,
    *,
    patient_id: UUID | None = None,
) -> None:
    require_consent(consent, OPERATION_SCOPE[operation], patient_id=patient_id)


def granted_scopes(consent: Consent | None) -> set[ConsentScope]:
    """The scopes currently in force — drives the consent-scoped PSG projection
    (docs/04 §5): fields outside these scopes are omitted, not nulled.
    """
    if consent is None or consent.revoked_at is not None:
        return set()
    return set(consent.scope)
