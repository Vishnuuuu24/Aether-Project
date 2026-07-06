"""LLM readiness probe: model-id parsing, match logic, auth header, unreachable."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ai.llm.config import GatewayConfig, LLMProfile
from ai.llm.readiness import ProbeResult, _extract_model_ids, _matches, probe


def _config(model: str = "qwen3.6-35b-a3b", api_key: str | None = None) -> GatewayConfig:
    return GatewayConfig(
        profile=LLMProfile.LOCAL,
        base_url="http://localhost:1234/v1",
        model=model,
        api_key=api_key,
    )


def test_extract_ids_handles_openai_vllm_and_bare_shapes() -> None:
    assert _extract_model_ids({"data": [{"id": "a"}, {"id": "b"}]}) == ["a", "b"]
    assert _extract_model_ids({"models": [{"model": "x"}]}) == ["x"]
    assert _extract_model_ids(["m1", "m2"]) == ["m1", "m2"]
    assert _extract_model_ids({"unexpected": 1}) == []


def test_matches_exact_and_substring_but_not_unset() -> None:
    assert _matches("qwen3.6-35b-a3b", ["qwen3.6-35b-a3b"]) is True
    # LM Studio may report the full repo id; config carries the short served id.
    assert _matches("qwen3.6-35b-a3b", ["mlx-community/Qwen3.6-35B-a3b-4bit-qwen3.6-35b-a3b"])
    assert _matches("qwen3.6-35b-a3b", ["some-other-model"]) is False
    assert _matches("unset", ["anything"]) is False
    assert _matches("", ["anything"]) is False


def test_probe_ready_when_expected_model_loaded() -> None:
    def fake_get(url: str, headers: Mapping[str, str], timeout: float) -> Any:
        assert url == "http://localhost:1234/v1/models"
        return {"data": [{"id": "qwen3.6-35b-a3b"}]}

    result = probe(_config(), http_get=fake_get)
    assert isinstance(result, ProbeResult)
    assert result.reachable is True
    assert result.model_ready is True
    assert result.loaded_models == ("qwen3.6-35b-a3b",)


def test_probe_reachable_but_wrong_model() -> None:
    result = probe(_config(), http_get=lambda u, h, t: {"data": [{"id": "gpt-oss-20b"}]})
    assert result.reachable is True
    assert result.model_ready is False
    assert "not in the loaded set" in result.detail


def test_probe_unset_model_cannot_verify() -> None:
    result = probe(_config(model="unset"), http_get=lambda u, h, t: {"data": []})
    assert result.reachable is True
    assert result.model_ready is False
    assert "unset" in result.detail


def test_probe_unreachable_returns_result_not_exception() -> None:
    def boom(url: str, headers: Mapping[str, str], timeout: float) -> Any:
        raise ConnectionError("connection refused")

    result = probe(_config(), http_get=boom)
    assert result.reachable is False
    assert result.model_ready is False
    assert result.detail.startswith("unreachable:")


def test_probe_sends_bearer_when_api_key_set() -> None:
    seen: dict[str, str] = {}

    def capture(url: str, headers: Mapping[str, str], timeout: float) -> Any:
        seen.update(headers)
        return {"data": [{"id": "qwen3.6-35b-a3b"}]}

    probe(_config(api_key="sk-lm-secret"), http_get=capture)
    assert seen["Authorization"] == "Bearer sk-lm-secret"


def test_probe_omits_auth_header_when_no_key() -> None:
    seen: dict[str, str] = {}

    def capture(url: str, headers: Mapping[str, str], timeout: float) -> Any:
        seen.update(headers)
        return {"data": []}

    probe(_config(), http_get=capture)
    assert "Authorization" not in seen
