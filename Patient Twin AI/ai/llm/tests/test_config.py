"""GatewayConfig.from_env: profile parsing, PHI policy, model-id resolution."""

from __future__ import annotations

from ai.llm.config import GatewayConfig, LLMProfile


def test_defaults_to_local_profile() -> None:
    cfg = GatewayConfig.from_env({})
    assert cfg.profile is LLMProfile.LOCAL
    assert cfg.profile.phi_allowed is True
    assert cfg.is_production_local() is True


def test_model_falls_back_to_primary_model() -> None:
    cfg = GatewayConfig.from_env({"PRIMARY_MODEL": "qwen3.6-35b-a3b"})
    assert cfg.model == "qwen3.6-35b-a3b"


def test_llm_model_overrides_primary_model() -> None:
    cfg = GatewayConfig.from_env({"PRIMARY_MODEL": "a", "LLM_MODEL": "b"})
    assert cfg.model == "b"


def test_external_profile_is_not_phi_allowed() -> None:
    cfg = GatewayConfig.from_env({"LLM_PROFILE": "external_deidentified"})
    assert cfg.profile.phi_allowed is False
    assert cfg.is_production_local() is False
