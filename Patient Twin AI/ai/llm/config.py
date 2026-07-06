"""LLM Gateway configuration (docs/06 §6, docs/08 §2b, CLAUDE.md).

One codebase, config-switched. Where inference runs is environment, not code:
  - `local`   → LM Studio / mlx-lm (Mac) or vLLM (server), self-hosted. PHI-allowed.
                Production patient traffic is HARD-PINNED here.
  - `external_deidentified` → OpenRouter etc. PHI-FORBIDDEN; payload must pass the
                de-identification filter (default-deny). Dev-only.
  - `dev`     → synthetic data only; never real patients.

Env (see .env.example):
    LLM_PROFILE=local|external_deidentified|dev
    LLM_BACKEND=mlx|vllm
    LLM_GATEWAY_BASE_URL=http://host.docker.internal:1234/v1
    PRIMARY_MODEL=<served model id>   (LLM_MODEL overrides it if set)
    OPENROUTER_API_KEY=<only for external_deidentified>
    LLM_GATEWAY_API_KEY=<optional bearer for a self-hosted server that requires auth,
                         e.g. an LM Studio 0.4.0 permission token>
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum


class LLMProfile(str, Enum):
    LOCAL = "local"
    EXTERNAL_DEIDENTIFIED = "external_deidentified"
    DEV = "dev"

    @property
    def phi_allowed(self) -> bool:
        """Only the self-hosted local profile may receive PHI (docs/06 §6)."""
        return self is LLMProfile.LOCAL


@dataclass(frozen=True)
class GatewayConfig:
    profile: LLMProfile
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 60.0
    temperature: float = 0.0  # deterministic-leaning; the LLM explains, it never decides

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> GatewayConfig:
        env = env if env is not None else os.environ
        profile = LLMProfile(env.get("LLM_PROFILE", "local"))
        # External egress uses OPENROUTER_API_KEY; a self-hosted server that turns on
        # auth (e.g. an LM Studio permission token) uses LLM_GATEWAY_API_KEY. Both are
        # sent as `Authorization: Bearer`; OpenRouter wins if somehow both are set.
        api_key = env.get("OPENROUTER_API_KEY") or env.get("LLM_GATEWAY_API_KEY") or None
        # One model identity: the version stamp (PRIMARY_MODEL) and the served API id
        # are the same thing. LLM_MODEL is an explicit override if they ever diverge.
        model = env.get("LLM_MODEL")
        if not model:
            model = env.get("PRIMARY_MODEL", "unset")
        return cls(
            profile=profile,
            base_url=env.get("LLM_GATEWAY_BASE_URL", "http://host.docker.internal:1234/v1"),
            model=model,
            api_key=api_key,
            timeout_seconds=float(env.get("LLM_TIMEOUT_SECONDS", "60")),
            temperature=float(env.get("LLM_TEMPERATURE", "0")),
        )

    def is_production_local(self) -> bool:
        return self.profile is LLMProfile.LOCAL
