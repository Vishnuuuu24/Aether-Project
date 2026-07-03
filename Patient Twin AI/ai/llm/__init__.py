"""v1 LLM Gateway (docs/06 §6, docs/10 T4.1).

`Gateway` implements the stable `LLMGateway` interface: it assembles a prompt from
a consent-scoped `PSGProjection` + retrieved evidence, calls a served model through
the `ChatClient` port with structured-output enforcement, and returns a
`ProposedOutput`. Profiles select backend + egress policy; `external_*` profiles run
a default-deny de-identification filter before anything leaves the trust boundary.

The model backend is behind the `ChatClient` port: `OpenAICompatClient` (httpx to an
OpenAI-compatible server — LM Studio/mlx-lm on Mac, vLLM on the server) is the real
adapter; deterministic fakes drive the tests, exactly like retrieval/coding.
"""

from .client import ChatClient, LLMUnavailable, OpenAICompatClient
from .config import GatewayConfig, LLMProfile
from .deid import DeidReport, EgressBlocked, scan_for_identifiers
from .gateway import GATEWAY_VERSION, Gateway

__all__ = [
    "GATEWAY_VERSION",
    "ChatClient",
    "DeidReport",
    "EgressBlocked",
    "Gateway",
    "GatewayConfig",
    "LLMProfile",
    "LLMUnavailable",
    "OpenAICompatClient",
    "scan_for_identifiers",
]
