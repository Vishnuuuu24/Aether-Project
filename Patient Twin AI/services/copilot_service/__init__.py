"""Copilot orchestration service (docs/07 §5, docs/10 T4.2).

Assembles the one legal path from a patient question to a user-facing answer:

    PSGProjection (in) → hybrid retrieve → LLM Gateway proposes → Policy decides → render

`Copilot.answer` always returns a Policy-approved `OutputContract` — including
abstained/suppressed/escalated cases. It NEVER returns raw LLM output, and every
approved claim carries an evidence ref (guaranteed by the Policy grounding gate). Side
effects (persist the output, enqueue clinician escalations, emit the audit event) go
through injected ports so the orchestration stays testable and deterministic.
"""

from .orchestrator import Copilot
from .ports import AuditSink, EscalationSink, OutputStore

__all__ = ["AuditSink", "Copilot", "EscalationSink", "OutputStore"]
