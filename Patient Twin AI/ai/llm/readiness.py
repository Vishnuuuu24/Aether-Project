"""LLM serving readiness probe (CLAUDE.md ops note; docs/08 §2b).

Operationalises the CLAUDE.md rule — *"confirm the MoE quant of the primary actually
loads ... before depending on it"* — as a read-only check the developer can run
before a demo, or a service can run at startup, instead of discovering a
missing/wrong model via a cryptic inference failure.

It lists models via the OpenAI-compatible `GET /v1/models`, which BOTH serving
backends expose (LM Studio's v1 server on the Mac, vLLM on the H200). That keeps the
probe portable and OUT of the generic `ChatClient` inference port (docs/02 §6).

Read-only by design. LM Studio 0.4.0 also ships native management endpoints
(`/api/v1/models/{load,unload,download}`), but we deliberately do NOT call them from
any runtime path: model lifecycle on the single-tenant Mac is an operator action,
and auto-loading could evict a model that is resident mid-session (docs/08 §2b).
"unreachable" and "wrong model" are normal answers here, so the probe returns a
result rather than raising — it is the thing doing the checking.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import GatewayConfig

# A GET returning parsed JSON; raises on transport/HTTP error. Injected in tests.
HttpGetJson = Callable[[str, Mapping[str, str], float], Any]

_UNSET_MODEL = "unset"


@dataclass(frozen=True)
class ProbeResult:
    reachable: bool
    loaded_models: tuple[str, ...]
    expected_model: str
    model_ready: bool
    detail: str


def _models_url(base_url: str) -> str:
    # base_url already ends in the OpenAI-compat root (…/v1) for both backends.
    return f"{base_url.rstrip('/')}/models"


def _extract_model_ids(data: Any) -> list[str]:
    """Pull model ids from the common listing shapes, defensively."""
    if isinstance(data, Mapping):
        items = data.get("data") or data.get("models") or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    ids: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            value = item.get("id") or item.get("model") or item.get("name")
            if value:
                ids.append(str(value))
        elif isinstance(item, str):
            ids.append(item)
    return ids


def _matches(expected: str, loaded: list[str]) -> bool:
    """A loaded model satisfies the expectation on an exact id match, or when one id
    is a suffix/substring of the other (LM Studio may report `owner/Repo-4bit` where
    the config carries the short served id, or vice-versa)."""
    if not expected or expected == _UNSET_MODEL:
        return False
    for model in loaded:
        if model == expected or expected in model or model in expected:
            return True
    return False


def _httpx_get_json(url: str, headers: Mapping[str, str], timeout: float) -> Any:
    import httpx  # local import: importing this module never opens a socket

    response = httpx.get(url, headers=dict(headers), timeout=timeout)
    response.raise_for_status()
    return response.json()


def probe(config: GatewayConfig, *, http_get: HttpGetJson | None = None) -> ProbeResult:
    """Check whether the configured model is loaded and serving at `config.base_url`."""
    get = http_get or _httpx_get_json
    headers = {"Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    try:
        data = get(_models_url(config.base_url), headers, config.timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - any transport failure means "not reachable"
        return ProbeResult(False, (), config.model, False, f"unreachable: {exc}")

    loaded = _extract_model_ids(data)
    ready = _matches(config.model, loaded)
    if config.model == _UNSET_MODEL:
        detail = "server reachable but PRIMARY_MODEL is unset — cannot verify"
    elif ready:
        detail = "ok: expected model is loaded"
    else:
        detail = f"server reachable but {config.model!r} is not in the loaded set"
    return ProbeResult(True, tuple(loaded), config.model, ready, detail)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Probe the LLM serving backend readiness.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="override LLM_GATEWAY_BASE_URL (host processes usually want localhost, "
        "not host.docker.internal)",
    )
    args = parser.parse_args(argv)

    config = GatewayConfig.from_env()
    if args.base_url:
        from dataclasses import replace

        config = replace(config, base_url=args.base_url)

    result = probe(config)
    print(f"base_url : {config.base_url}")
    print(f"reachable: {result.reachable}")
    print(f"loaded   : {list(result.loaded_models)}")
    print(f"expected : {result.expected_model}")
    print(f"ready    : {result.model_ready}  ({result.detail})")
    if not result.reachable:
        print(
            "hint: a process on the Mac host (not in Docker) must use "
            "http://localhost:1234/v1 — host.docker.internal only resolves in containers."
        )
    return 0 if result.model_ready else 1


if __name__ == "__main__":
    raise SystemExit(_main())
