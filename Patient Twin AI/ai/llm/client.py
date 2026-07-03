"""`ChatClient` port + the real OpenAI-compatible adapter (docs/03, docs/08 §2b).

The gateway depends on this narrow port, not on any specific serving stack. The real
adapter (`OpenAICompatClient`) talks to any OpenAI-compatible `/chat/completions`
endpoint — LM Studio / mlx-lm on the Mac (port 1234), vLLM on the server — and asks
for structured output via `response_format: json_schema`. Tests inject a deterministic
fake implementing the same port, so no live model is needed to exercise the gateway.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class LLMUnavailable(Exception):
    """The served model could not be reached or returned an unusable response.

    Per docs/06 §9 the copilot must treat this as an abstention — never fall back to
    ungrounded generation.
    """


@runtime_checkable
class ChatClient(Protocol):
    def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        schema_name: str,
        temperature: float,
    ) -> str:
        """Return the model's response as a JSON string constrained to `json_schema`.

        Implementations MUST raise `LLMUnavailable` on transport/HTTP failure rather
        than returning partial or free-text content.
        """
        ...


class OpenAICompatClient:
    """httpx adapter for an OpenAI-compatible chat endpoint. Lazy — importing this
    module never opens a socket; the client is created on first use, mirroring the
    lazy real adapters in retrieval/coding.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        schema_name: str,
        temperature: float,
    ) -> str:
        import httpx  # local import: keep module import side-effect free

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
            },
        }
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMUnavailable(str(exc)) from exc
