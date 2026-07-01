"""Single source of contracts. Every service imports from here; no service
redefines a contract locally (see CLAUDE.md "Hard rules for the agent").
"""

from .audit import AuditAction, AuditActor, AuditRecord
from .consent import Consent, ConsentScope
from .output_contract import (
    MANDATORY_DISCLAIMER,
    Evidence,
    EvidenceKind,
    OutputContract,
    OutputType,
    PolicyDecision,
    PolicyRecord,
    ProposedOutput,
    RecommendedAction,
    Severity,
)
from .patient import PatientProfile, SexAtBirth
from .psg import (
    AllergyNode,
    BaselineNode,
    ConditionNode,
    DeviationDirection,
    DeviationNode,
    EventNode,
    EventSeverity,
    ForecastNode,
    MedicationNode,
    ObservationNode,
    PSGProjection,
    VersionedNode,
)
from .reading import (
    REQUIRED_CORE_METRICS,
    IngestBatchResult,
    IngestRejection,
    MeasurementContext,
    MetricCode,
    Reading,
)
from .vector import VectorPayload, VectorSourceType

__all__ = [
    "AuditAction",
    "AuditActor",
    "AuditRecord",
    "Consent",
    "ConsentScope",
    "PatientProfile",
    "SexAtBirth",
    "MANDATORY_DISCLAIMER",
    "Evidence",
    "EvidenceKind",
    "OutputContract",
    "OutputType",
    "PolicyDecision",
    "PolicyRecord",
    "ProposedOutput",
    "RecommendedAction",
    "Severity",
    "AllergyNode",
    "BaselineNode",
    "ConditionNode",
    "DeviationDirection",
    "DeviationNode",
    "EventNode",
    "EventSeverity",
    "ForecastNode",
    "MedicationNode",
    "ObservationNode",
    "PSGProjection",
    "VersionedNode",
    "REQUIRED_CORE_METRICS",
    "IngestBatchResult",
    "IngestRejection",
    "MeasurementContext",
    "MetricCode",
    "Reading",
    "VectorPayload",
    "VectorSourceType",
]
