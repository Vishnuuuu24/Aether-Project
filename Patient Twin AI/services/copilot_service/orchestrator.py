"""Copilot orchestrator (docs/07 §5, docs/10 T4.2).

`answer()` is the only sanctioned route from a question to a user-facing output. It is
thin by design — the intelligence lives in the deterministic engines it wires
together, not here:

    retrieve (consent-scoped) → gateway.propose → policy.decide → persist/audit/escalate

Guarantees it upholds:
  * Raw LLM output never leaves this method — only a Policy-approved OutputContract.
  * Gateway failure / blocked egress → a Policy-issued abstention, never ungrounded
    fallback generation (docs/06 §9).
  * Every returned contract carries a policy decision record and is persisted + audited;
    escalations are enqueued for clinician review.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from ai.interfaces.llm_gateway import LLMGateway
from ai.interfaces.retriever import Retriever
from ai.llm.client import LLMUnavailable
from ai.llm.deid import EgressBlocked
from core.versioning.registry import VersionSet
from schemas.consent import ConsentScope
from schemas.output_contract import OutputContract
from schemas.psg import PSGProjection
from schemas.retrieval import RetrievalScope
from services.policy_engine.engine import PolicyEngine

from .ports import (
    AuditSink,
    EscalationSink,
    NullAuditSink,
    NullEscalationSink,
    NullOutputStore,
    OutputStore,
)

COPILOT_VERSION = "copilot-v1"

_DEFAULT_K = 8


class Copilot:
    def __init__(
        self,
        *,
        retriever: Retriever,
        gateway: LLMGateway,
        policy: PolicyEngine,
        versions: VersionSet,
        k: int = _DEFAULT_K,
        output_store: OutputStore | None = None,
        escalation_sink: EscalationSink | None = None,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self._retriever = retriever
        self._gateway = gateway
        self._policy = policy
        self._versions = versions
        self._k = k
        self._output_store = output_store or NullOutputStore()
        self._escalation = escalation_sink or NullEscalationSink()
        self._audit = audit_sink or NullAuditSink()

    def answer(
        self,
        *,
        patient_id: UUID,
        projection: PSGProjection,
        query: str,
        consented_scopes: list[ConsentScope],
        now: datetime,
        locale: str = "en",
    ) -> OutputContract:
        scope = RetrievalScope(
            patient_id=patient_id, consented_scopes=consented_scopes, include_kb=True
        )
        evidence = self._retriever.search(query, scope, k=self._k)

        try:
            proposal = self._gateway.propose(
                query=query, projection=projection, evidence=evidence, locale=locale
            )
        except EgressBlocked as exc:
            output = self._policy.on_gateway_failure(
                f"request could not be de-identified for the configured model ({exc})",
                projection=projection,
                patient_id=patient_id,
                versions=self._versions,
                now=now,
            )
        except LLMUnavailable as exc:
            output = self._policy.on_gateway_failure(
                f"the assistant is temporarily unavailable ({exc})",
                projection=projection,
                patient_id=patient_id,
                versions=self._versions,
                now=now,
            )
        else:
            output = self._policy.decide(
                proposal,
                projection,
                patient_id=patient_id,
                evidence=evidence,
                versions=self._versions,
                now=now,
            )

        self._emit(output)
        return output

    def _emit(self, output: OutputContract) -> None:
        # Persist + audit every output (incl. abstain/suppress); escalate red flags.
        self._output_store.save(output)
        self._audit.record(output)
        if output.escalation.triggered:
            self._escalation.enqueue(output)
