"""The authenticated caller, distilled from verified JWT claims (docs/07 §1).

A Principal is *who is asking*. It is deliberately separate from consent
(*what the patient permitted*): RBAC uses the Principal, the consent gate uses
the patient's `Consent` record. Both must pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from uuid import UUID


class Role(str, Enum):
    """Maps onto audit actors (docs/04 §7) plus an admin/governance role."""

    PATIENT = "patient"
    CLINICIAN = "clinician"
    SYSTEM = "system"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    subject: str  # pseudonymous `sub` claim — never a raw identifier
    roles: frozenset[Role]
    token_scopes: frozenset[str] = frozenset()
    patient_id: UUID | None = None

    def has_role(self, role: Role) -> bool:
        return role in self.roles

    @property
    def is_system(self) -> bool:
        return Role.SYSTEM in self.roles
