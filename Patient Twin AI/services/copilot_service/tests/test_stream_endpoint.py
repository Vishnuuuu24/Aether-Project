"""POST /v1/patients/{id}/copilot/query:stream — SSE variant (docs/07 §5; docs/15 T6.5).

The deterministic Policy decision runs to completion BEFORE any bytes stream, so the
SSE body can only ever carry vetted content. Approved/downgraded answers stream their
message tokens; abstained/suppressed decisions stream no token content.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from core.auth.errors import ConsentError
from schemas.output_contract import (
    Abstention,
    Escalation,
    Evidence,
    EvidenceKind,
    OutputContract,
    OutputType,
    PolicyDecision,
    PolicyRecord,
    RecommendedAction,
    Severity,
    VersionStamp,
)
from services.copilot_service.app.main import app, get_copilot, get_state_engine
from services.patient_state_engine.service import ProfileNotFoundError

from ._helpers import NOW, make_projection

_VERSIONS = VersionStamp(model="m1", ruleset="r1", baseline_engine="b1", prompt="p1")
_APPROVED_MESSAGE = "Your resting heart rate rose above your usual range overnight."


class _FakeEngine:
    def __init__(self, projection: Any = None, error: Exception | None = None) -> None:
        self._projection = projection
        self._error = error

    def build_projection(self, patient_id: Any) -> Any:
        if self._error is not None:
            raise self._error
        return self._projection


class _FakeCopilot:
    def __init__(self, output: OutputContract) -> None:
        self._output = output

    def answer(self, **kwargs: Any) -> OutputContract:
        return self._output


def _approved(patient_id: Any) -> OutputContract:
    return OutputContract(
        patient_id=patient_id,
        type=OutputType.INFO,
        message=_APPROVED_MESSAGE,
        severity=Severity.LOW,
        confidence=0.7,
        evidence=[Evidence(kind=EvidenceKind.KB_PASSAGE, ref="kb:1", quote_or_fact="RHR rises.")],
        recommended_action=RecommendedAction.MONITOR,
        escalation=Escalation(),
        abstained=Abstention(),
        policy=PolicyRecord(decision=PolicyDecision.APPROVED, rule_ids=["R2_grounding"]),
        versions=_VERSIONS,
        created_at=NOW,
    )


def _abstained(patient_id: Any) -> OutputContract:
    return OutputContract(
        patient_id=patient_id,
        type=OutputType.INFO,
        message="I can't answer this safely from the available data.",
        severity=Severity.NONE,
        confidence=0.0,
        evidence=[],
        recommended_action=RecommendedAction.NONE,
        abstained=Abstention(value=True, reason="insufficient grounded evidence"),
        policy=PolicyRecord(decision=PolicyDecision.ABSTAIN, rule_ids=["R2_grounding"]),
        versions=_VERSIONS,
        created_at=NOW,
    )


def _client(engine: _FakeEngine, copilot: _FakeCopilot) -> TestClient:
    app.dependency_overrides[get_state_engine] = lambda: engine
    app.dependency_overrides[get_copilot] = lambda: copilot
    return TestClient(app)


def _events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for block in text.strip().split("\n\n"):
        event: dict[str, Any] = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                event["event"] = line[len("event: ") :]
            elif line.startswith("data: "):
                event["data"] = json.loads(line[len("data: ") :])
        if event:
            events.append(event)
    return events


def test_stream_approved_streams_vetted_tokens_then_result() -> None:
    pid = uuid4()
    client = _client(_FakeEngine(projection=make_projection()), _FakeCopilot(_approved(pid)))
    try:
        resp = client.post(f"/v1/patients/{pid}/copilot/query:stream", json={"query": "why?"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _events(resp.text)

    tokens = [e for e in events if e["event"] == "token"]
    assert tokens  # a real answer streams its message tokens
    # The streamed tokens are exactly the vetted message (no raw LLM content).
    assert " ".join(t["data"]["text"] for t in tokens) == " ".join(_APPROVED_MESSAGE.split())

    results = [e for e in events if e["event"] == "result"]
    assert len(results) == 1
    assert results[0]["data"]["policy"]["decision"] == "approved"
    assert results[0]["data"]["evidence"]  # full structured contract carried
    assert events[-1]["event"] == "done"


def test_stream_abstained_streams_no_token_content() -> None:
    pid = uuid4()
    try:
        resp = _client(
            _FakeEngine(projection=make_projection()), _FakeCopilot(_abstained(pid))
        ).post(f"/v1/patients/{pid}/copilot/query:stream", json={"query": "why?"})
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 200
    events = _events(resp.text)
    assert [e for e in events if e["event"] == "token"] == []  # nothing un-vetted streamed
    result = next(e for e in events if e["event"] == "result")
    assert result["data"]["abstained"]["value"] is True


def test_stream_403_without_consent_is_not_a_stream() -> None:
    pid = uuid4()
    try:
        resp = _client(
            _FakeEngine(error=ConsentError("no consent in force")), _FakeCopilot(_approved(pid))
        ).post(f"/v1/patients/{pid}/copilot/query:stream", json={"query": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 403


def test_stream_404_unknown_patient() -> None:
    pid = uuid4()
    try:
        resp = _client(
            _FakeEngine(error=ProfileNotFoundError("unknown")), _FakeCopilot(_approved(pid))
        ).post(f"/v1/patients/{pid}/copilot/query:stream", json={"query": "hi"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 404
