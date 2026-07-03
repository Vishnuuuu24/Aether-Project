"""Stable `LLMGateway` interface (docs/02 §6, docs/06 §6).

The gateway is the ONLY component that talks to a language model. It takes a
consent-scoped `PSGProjection` plus retrieved evidence and returns a *proposed*
structured output — never a user-facing one. Contract with the rest of the system:

  - The LLM never sees raw physiological signals (CLAUDE.md principle 2). The input
    is a `PSGProjection`, which by construction carries no reading-level data.
  - Output is a `ProposedOutput` (schemas/output_contract), never an `OutputContract`.
    Only the Policy Engine may turn a proposal into an approved user-facing output.
  - Profiles (`local` / `external_deidentified` / `dev`) select the backend and the
    egress rules. Production patient traffic is hard-pinned to `local`; `external_*`
    is PHI-forbidden and default-deny (docs/06 §6).

Deferred swaps (a different served model, a different serving stack) are new
implementations of the `ChatClient` port behind this gateway, never a new call site.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from schemas.output_contract import ProposedOutput
from schemas.psg import PSGProjection
from schemas.retrieval import EvidenceChunk


@runtime_checkable
class LLMGateway(Protocol):
    def propose(
        self,
        *,
        query: str,
        projection: PSGProjection,
        evidence: list[EvidenceChunk],
        locale: str = "en",
    ) -> ProposedOutput: ...
