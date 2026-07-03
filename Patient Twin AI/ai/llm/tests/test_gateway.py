"""Gateway: structured output, profile egress rules, fail-safe behaviour
(docs/06 §6, §9; docs/10 T4.1)."""

from __future__ import annotations

import pytest

from ai.grounding import allowed_refs, kb_ref
from ai.llm.client import LLMUnavailable
from ai.llm.config import GatewayConfig, LLMProfile
from ai.llm.deid import EgressBlocked
from ai.llm.gateway import Gateway
from schemas.output_contract import ProposedOutput

from ._fixtures import KB_CHUNK_ID, FakeChatClient, make_evidence, make_projection


def _local_config() -> GatewayConfig:
    return GatewayConfig(profile=LLMProfile.LOCAL, base_url="http://x/v1", model="m1")


def _valid_proposal_json() -> str:
    proposal = ProposedOutput.model_validate(
        {
            "type": "info",
            "message": "Your resting heart rate rose above your personal baseline.",
            "severity": "low",
            "confidence": 0.7,
            "evidence": [
                {
                    "kind": "kb_passage",
                    "ref": f"kb:{KB_CHUNK_ID}",
                    "quote_or_fact": "RHR rises with acute stress or illness.",
                }
            ],
            "recommended_action": "monitor",
        }
    )
    return proposal.model_dump_json()


def test_valid_structured_response_is_parsed() -> None:
    client = FakeChatClient(response=_valid_proposal_json())
    gw = Gateway(_local_config(), client)
    out = gw.propose(
        query="why is my HR up?", projection=make_projection(), evidence=make_evidence()
    )
    assert isinstance(out, ProposedOutput)
    assert out.type.value == "info"
    # The prompt handed the model exactly the refs the grounding gate will accept.
    assert kb_ref(make_evidence()[0]) in allowed_refs(make_projection(), make_evidence())


def test_transport_failure_raises_unavailable() -> None:
    client = FakeChatClient(raises=LLMUnavailable("connection refused"))
    gw = Gateway(_local_config(), client)
    with pytest.raises(LLMUnavailable):
        gw.propose(query="q", projection=make_projection(), evidence=make_evidence())


def test_offschema_response_raises_unavailable() -> None:
    client = FakeChatClient(response='{"not":"a valid proposal"}')
    gw = Gateway(_local_config(), client)
    with pytest.raises(LLMUnavailable):
        gw.propose(query="q", projection=make_projection(), evidence=make_evidence())


def test_local_profile_allows_phi_in_query() -> None:
    # 'local' is self-hosted and PHI-allowed: an identifier in the query must NOT block.
    client = FakeChatClient(response=_valid_proposal_json())
    gw = Gateway(_local_config(), client)
    out = gw.propose(
        query="I'm John, DOB 1984-02-11, why is my HR up?",
        projection=make_projection(),
        evidence=make_evidence(),
    )
    assert isinstance(out, ProposedOutput)


def test_external_profile_blocks_phi_egress() -> None:
    config = GatewayConfig(
        profile=LLMProfile.EXTERNAL_DEIDENTIFIED,
        base_url="http://x/v1",
        model="m1",
        api_key="k",
    )
    client = FakeChatClient(response=_valid_proposal_json())
    gw = Gateway(config, client)
    with pytest.raises(EgressBlocked):
        gw.propose(
            query="my email is a@b.com, explain my HR",
            projection=make_projection(),
            evidence=make_evidence(),
        )
    assert client.last_user is None  # nothing was ever sent to the model


def test_external_profile_requires_api_key() -> None:
    config = GatewayConfig(
        profile=LLMProfile.EXTERNAL_DEIDENTIFIED, base_url="http://x/v1", model="m1"
    )
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Gateway(config, FakeChatClient(response="{}"))


def test_non_projection_context_refused() -> None:
    gw = Gateway(_local_config(), FakeChatClient(response=_valid_proposal_json()))
    with pytest.raises(TypeError):
        gw.propose(query="q", projection={"not": "a projection"}, evidence=[])  # type: ignore[arg-type]


def test_phi_allowed_property() -> None:
    assert LLMProfile.LOCAL.phi_allowed is True
    assert LLMProfile.EXTERNAL_DEIDENTIFIED.phi_allowed is False
    assert LLMProfile.DEV.phi_allowed is False
