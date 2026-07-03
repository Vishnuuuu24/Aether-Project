"""Service-local ports for the copilot's side effects (docs/06 §7-8, docs/07 §6).

Not part of the canonical ai/interfaces set — these are swappable sinks the
orchestrator writes to after the Policy Engine has decided:

  - `OutputStore`     persists every OutputContract (docs/04 §6: any answer is
                      reconstructable) — real impl writes `core.db.models.OutputRecord`.
  - `EscalationSink`  enqueues red-flag / high-severity outputs for clinician review
                      (docs/06 §7, docs/07 §6). v1 is a read-only queue.
  - `AuditSink`       emits the hash-chained audit event for the decision (docs/06 §8)
                      — real impl wraps `core.audit`.

Default no-op implementations let the orchestrator run (and be unit-tested) without a
database; the real adapters are wired at the API/service boundary.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas.output_contract import OutputContract


@runtime_checkable
class OutputStore(Protocol):
    def save(self, output: OutputContract) -> None: ...


@runtime_checkable
class EscalationSink(Protocol):
    def enqueue(self, output: OutputContract) -> None: ...


@runtime_checkable
class AuditSink(Protocol):
    def record(self, output: OutputContract) -> None: ...


class NullOutputStore:
    def save(self, output: OutputContract) -> None:  # noqa: D401 - no-op default
        return None


class NullEscalationSink:
    def enqueue(self, output: OutputContract) -> None:
        return None


class NullAuditSink:
    def record(self, output: OutputContract) -> None:
        return None
