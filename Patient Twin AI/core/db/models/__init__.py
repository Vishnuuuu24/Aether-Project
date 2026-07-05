"""SQLAlchemy ORM models — the relational realisation of `schemas/` (docs/04 §3).

These tables ARE the relational Patient State Graph plus the governance tables.
The pydantic models in `schemas/` remain the single source of the *shape*; these
ORM models are its storage projection and must not diverge from it. Enums are
stored as strings (validated at the pydantic layer) to keep migrations cheap.

Organised by domain for tracking & maintainability:

    common.py      — NOW default + VersionedMixin
    governance.py  — patient_profile, consent, audit_log, version_registry
    readings.py    — reading_node
    psg.py         — versioned PSG nodes (baseline/deviation/event/condition/…)
    outputs.py     — output

EVERY model is re-exported here, which matters for alembic: env.py does
`from core.db import models`, so importing this package registers all tables on
`Base.metadata` (autogenerate & `alembic check` see the full schema). It also
keeps `from core.db.models import AuditLog` working. **When you add a model,
import it here** or alembic won't see it.
"""
from __future__ import annotations

from .common import NOW, VersionedMixin
from .governance import AuditLog, Consent, Outcome, PatientProfile, VersionRegistryRow
from .outputs import OutputRecord
from .psg import (
    AllergyNode,
    BaselineNode,
    ConditionNode,
    DeviationNode,
    DocumentNode,
    EventNode,
    ForecastNode,
    MedicationNode,
    ObservationNode,
)
from .readings import ReadingNode

__all__ = [
    "NOW",
    "VersionedMixin",
    # governance
    "PatientProfile",
    "Consent",
    "AuditLog",
    "Outcome",
    "VersionRegistryRow",
    # readings
    "ReadingNode",
    # psg nodes
    "BaselineNode",
    "DeviationNode",
    "EventNode",
    "ConditionNode",
    "MedicationNode",
    "AllergyNode",
    "ObservationNode",
    "ForecastNode",
    "DocumentNode",
    # outputs
    "OutputRecord",
]
