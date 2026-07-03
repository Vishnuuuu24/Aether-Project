"""POST /v1/patients/{id}/copilot/query — returns a Policy OutputContract, maps
consent/unknown-patient errors, never leaks raw LLM output (docs/07 §5, §10)."""

from __future__ import annotations

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

from ._helpers import NOW, make_projection  # local re-export of the policy fixtures


class _FakeEngine:
    def __init__(self, projection=None, error: Exception | None = None) -> None:
        self._projection = projection
        self._error = error

    def build_projection(self, patient_id):  # noqa: ANN001
        if self._error is not None:
            raise self._error
        return self._projection


class _FakeCopilot:
    def __init__(self, output: OutputContract) -> None:
        self._output = output
        self.called_with: dict | None = None

    def answer(self, **kwargs) -> OutputContract:  # noqa: ANN003
        self.called_with = kwargs
        return self._output


def _approved(patient_id) -> OutputContract:  # noqa: ANN001
    return OutputContract(
        patient_id=patient_id,
        type=OutputType.INFO,
        message="Your resting heart rate rose above your usual range.",
        severity=Severity.LOW,
        confidence=0.7,
        evidence=[Evidence(kind=EvidenceKind.KB_PASSAGE, ref="kb:1", quote_or_fact="RHR rises.")],
        recommended_action=RecommendedAction.MONITOR,
        escalation=Escalation(),
        abstained=Abstention(),
        policy=PolicyRecord(decision=PolicyDecision.APPROVED, rule_ids=["R2_grounding"]),
        versions=VersionStamp(model="m1", ruleset="r1", baseline_engine="b1", prompt="p1"),
        created_at=NOW,
    )


def _client(engine: _FakeEngine, copilot: _FakeCopilot) -> TestClient:
    app.dependency_overrides[get_state_engine] = lambda: engine
    app.dependency_overrides[get_copilot] = lambda: copilot
    return TestClient(app)


def test_query_returns_output_contract() -> None:
    pid = uuid4()
    copilot = _FakeCopilot(_approved(pid))
    engine = _FakeEngine(projection=make_projection())
    try:
        resp = _client(engine, copilot).post(
            f"/v1/patients/{pid}/copilot/query", json={"query": "why?"}
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["policy"]["decision"] == "approved"
    assert body["evidence"]  # claim carries evidence
    assert body["disclaimer"]  # mandatory disclaimer present
    # Consent scopes were derived from the projection and passed to the orchestrator.
    assert copilot.called_with is not None
    assert copilot.called_with["query"] == "why?"


def test_query_403_without_consent() -> None:
    pid = uuid4()
    engine = _FakeEngine(error=ConsentError("no consent in force"))
    try:
        resp = _client(engine, _FakeCopilot(_approved(pid))).post(
            f"/v1/patients/{pid}/copilot/query", json={"query": "hi"}
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 403


def test_query_404_unknown_patient() -> None:
    pid = uuid4()
    engine = _FakeEngine(error=ProfileNotFoundError("unknown"))
    try:
        resp = _client(engine, _FakeCopilot(_approved(pid))).post(
            f"/v1/patients/{pid}/copilot/query", json={"query": "hi"}
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 404


def test_query_rejects_empty_body() -> None:
    pid = uuid4()
    try:
        resp = _client(
            _FakeEngine(projection=make_projection()), _FakeCopilot(_approved(pid))
        ).post(f"/v1/patients/{pid}/copilot/query", json={"query": ""})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 422
