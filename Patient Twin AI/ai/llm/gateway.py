"""LLM Gateway v1 (docs/06 §6, docs/10 T4.1).

Implements the stable `LLMGateway` interface. Flow:

    PSGProjection + evidence + query
        -> assemble (system, user) prompt (ai.llm.prompt)
        -> [external_* profiles only] default-deny de-identification gate
        -> ChatClient.complete_json  (structured output: ProposedOutput schema)
        -> parse into ProposedOutput  (never an OutputContract — Policy owns that)

Failure is fail-safe: any transport error, unparseable response, or blocked egress
raises, and the copilot turns that into an abstention (docs/06 §9) — the gateway
NEVER returns free-text or ungrounded content.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from schemas.output_contract import ProposedOutput
from schemas.psg import PSGProjection
from schemas.retrieval import EvidenceChunk

from .client import ChatClient, LLMUnavailable, OpenAICompatClient
from .config import GatewayConfig, LLMProfile
from .deid import assert_clean_for_egress
from .prompt import SYSTEM_PROMPT, build_user_message

GATEWAY_VERSION = "llm-gateway-v1"

_SCHEMA_NAME = "proposed_output"


class Gateway:
    """Concrete `LLMGateway`. Construct with an explicit `ChatClient` (tests inject a
    fake); `from_config` builds the real OpenAI-compatible client for LM Studio/vLLM.
    """

    def __init__(self, config: GatewayConfig, client: ChatClient) -> None:
        self._config = config
        self._client = client
        if config.profile is LLMProfile.EXTERNAL_DEIDENTIFIED and not config.api_key:
            # A dev-only external profile with no key is a misconfiguration, not a
            # runtime abstention — fail loudly at construction.
            raise ValueError("external_deidentified profile requires OPENROUTER_API_KEY")

    @classmethod
    def from_config(cls, config: GatewayConfig | None = None) -> Gateway:
        config = config or GatewayConfig.from_env()
        client = OpenAICompatClient(
            base_url=config.base_url,
            model=config.model,
            api_key=config.api_key,
            timeout_seconds=config.timeout_seconds,
        )
        return cls(config, client)

    @property
    def version(self) -> str:
        return GATEWAY_VERSION

    def propose(
        self,
        *,
        query: str,
        projection: PSGProjection,
        evidence: list[EvidenceChunk],
        locale: str = "en",
    ) -> ProposedOutput:
        # Defense in depth: the LLM only ever sees a projection (no raw signals).
        if not isinstance(projection, PSGProjection):
            raise TypeError("LLM Gateway refuses any context that is not a PSGProjection")

        user = build_user_message(
            query=query, projection=projection, evidence=evidence, locale=locale
        )

        # PHI never leaves the trust boundary: external profiles are default-deny.
        if not self._config.profile.phi_allowed:
            assert_clean_for_egress(f"{SYSTEM_PROMPT}\n{user}")

        raw = self._client.complete_json(
            system=SYSTEM_PROMPT,
            user=user,
            json_schema=_proposed_output_schema(),
            schema_name=_SCHEMA_NAME,
            temperature=self._config.temperature,
        )
        try:
            return ProposedOutput.model_validate_json(raw)
        except ValidationError as exc:
            # An unparseable / off-schema response is unusable — treat as unavailable
            # so the copilot abstains rather than shipping malformed content.
            raise LLMUnavailable(f"model response failed schema validation: {exc}") from exc


def _proposed_output_schema() -> dict[str, Any]:
    return ProposedOutput.model_json_schema()
