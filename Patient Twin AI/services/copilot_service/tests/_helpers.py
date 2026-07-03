"""Re-export the policy engine's deterministic fixtures for copilot tests (DRY)."""

from __future__ import annotations

from services.policy_engine.tests._fixtures import (
    NOW,
    PATIENT_ID,
    VERSIONS,
    grounded_proposal,
    kb_evidence,
    make_projection,
)

__all__ = [
    "NOW",
    "PATIENT_ID",
    "VERSIONS",
    "grounded_proposal",
    "kb_evidence",
    "make_projection",
]
