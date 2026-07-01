"""T0.1 DoD: 'OpenAPI generates'. The contracts must assemble into a valid
OpenAPI document and each must emit a JSON Schema.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from schemas import (
    AuditRecord,
    Consent,
    OutputContract,
    PatientProfile,
    PSGProjection,
    Reading,
    VectorPayload,
)

CONTRACTS: list[type[BaseModel]] = [
    PatientProfile,
    Consent,
    Reading,
    VectorPayload,
    PSGProjection,
    OutputContract,
    AuditRecord,
]


def build_app() -> FastAPI:
    app = FastAPI(title="patient-copilot-contracts")

    @app.post("/patients", response_model=PatientProfile)
    def create_patient(profile: PatientProfile) -> PatientProfile:
        return profile

    @app.post("/readings", response_model=Reading)
    def ingest_reading(reading: Reading) -> Reading:
        return reading

    @app.post("/vectors", response_model=VectorPayload)
    def upsert_vector(payload: VectorPayload) -> VectorPayload:
        return payload

    @app.get("/state", response_model=PSGProjection)
    def get_state() -> PSGProjection:  # pragma: no cover - never executed
        raise NotImplementedError

    @app.post("/outputs", response_model=OutputContract)
    def emit_output(output: OutputContract) -> OutputContract:
        return output

    return app


def test_openapi_document_generates() -> None:
    spec = build_app().openapi()
    assert spec["openapi"].startswith("3.")
    component_schemas = spec["components"]["schemas"]
    # FastAPI/pydantic may split a model used as both request body and response
    # into `<Name>-Input` / `<Name>-Output`; accept either form.
    for name in ("PatientProfile", "Reading", "VectorPayload", "PSGProjection", "OutputContract"):
        assert any(
            key == name or key.startswith(f"{name}-") for key in component_schemas
        ), f"{name} missing from OpenAPI components"


def test_openapi_endpoint_serves() -> None:
    # The generated spec is also reachable over HTTP (FastAPI's /openapi.json).
    client = TestClient(build_app())
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"] == "patient-copilot-contracts"


def test_every_contract_emits_json_schema() -> None:
    for model in CONTRACTS:
        schema = model.model_json_schema()
        assert schema.get("type") == "object"
        assert "properties" in schema
