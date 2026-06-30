# 07 — API Specification

REST over HTTPS, JSON, FastAPI. All endpoints require a valid JWT; access is RBAC- and consent-scoped. Versioned under `/v1`. Patient IDs are UUID7. Errors use RFC 7807 problem+json.

## 1. Auth & conventions

- `Authorization: Bearer <jwt>`; claims include `sub` (pseudonymous), `roles`, `scopes`.
- Idempotency: ingestion endpoints accept `Idempotency-Key`.
- Pagination: cursor-based (`?cursor=&limit=`).
- Every response includes `X-Trace-Id`; every mutating call is audited.

## 2. Consent

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/patients/{id}/consent` | Grant/update scoped consent (version, scope[]) |
| DELETE | `/v1/patients/{id}/consent` | Revoke (stops all processing in scope) |
| GET | `/v1/patients/{id}/consent` | Current consent record |

No other endpoint proceeds without a valid covering consent.

## 3. Ingestion

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/v1/ingest/readings` | `Reading[]` (`04 §2`) | Batch; per-item 207 multi-status; rejects on missing metadata |
| POST | `/v1/ingest/documents` | multipart (file + `doc_type`) | → OCR→coding→FHIR (proposed) |
| POST | `/v1/ingest/adapters/{adapter}/webhook` | adapter payload | HealthKit/Health Connect/Fitbit push |
| POST | `/v1/ingest/replay` | `{dataset, patient_id, speed}` | dev-only dataset replay harness |

## 4. Twin / state (read)

| Method | Path | Returns |
|---|---|---|
| GET | `/v1/patients/{id}/state` | PSG projection (`04 §5`), consent-scoped |
| GET | `/v1/patients/{id}/baselines?metric=&context=` | Baseline nodes (+ `is_population_fallback`) |
| GET | `/v1/patients/{id}/deviations?since=` | Recent deviations |
| GET | `/v1/patients/{id}/events?status=` | Events (severity, contributing deviations) |
| GET | `/v1/patients/{id}/forecast?metric=&horizon=` | Forecast nodes |
| GET | `/v1/patients/{id}/observations?code=` | Coded observations from documents |
| GET | `/v1/patients/{id}/documents` | Document references + coding status |

## 5. Copilot

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/v1/patients/{id}/copilot/query` | `{query, locale}` | **Output contract** (`04 §6`), post-Policy only |

Behaviour: assemble PSG projection → hybrid retrieve → LLM Gateway proposes → Policy Engine decides → render. Always returns a valid output contract, including `abstained`/`suppressed` cases. Streaming variant `/copilot/query:stream` (SSE) streams only **after** Policy approval of the structured plan (no streaming of un-vetted content).

## 6. Clinician escalation (read-only stub, v1)

| Method | Path | Returns |
|---|---|---|
| GET | `/v1/escalations?status=open` | Queued red-flag/high-severity outputs |
| POST | `/v1/escalations/{id}/ack` | Clinician acknowledgement (audited) |

## 7. Governance / admin

| Method | Path | Purpose |
|---|---|---|
| GET | `/v1/audit?patient_id=&action=&since=` | Audit trail (hash-chained) |
| GET | `/v1/versions` | Active model/ruleset/prompt/schema versions |
| POST | `/v1/outcomes` | Record real clinical outcome for outer-loop capture |

## 8. Health & ops

`GET /healthz` (liveness), `GET /readyz` (deps: Postgres, Qdrant, vLLM, Redis), `GET /metrics` (Prometheus).

## 9. Error contract (problem+json)

```json
{"type":"about:blank","title":"string","status":422,
 "detail":"string","instance":"/v1/...","errors":[{"field":"timestamp","issue":"missing timezone"}]}
```

## 10. Rate limits & safety

- Per-patient and per-token rate limits at the gateway.
- The copilot endpoint enforces the deterministic Policy Engine server-side; clients cannot bypass it. No endpoint returns raw LLM output.
