# 14 — Hardening Runbook (T5.3)

Operational status of the production-hardening checklist (`08 §6`), the NFR
measurements, and the failure-mode verification. This is the T5.3 record:
**NFRs measured (or explicitly deferred to the H200); failure-mode defaults
verified.**

## 1. Failure-mode defaults (`06 §9`) — VERIFIED

Every row of the `06 §9` table has an end-to-end acceptance test through the real
Copilot orchestrator + Policy Engine + de-id gate:
`services/copilot_service/tests/test_failure_modes.py`.

| Failure | Default behaviour | Verified by |
|---|---|---|
| LLM Gateway unavailable | abstain; never ungrounded generation | `test_llm_gateway_unavailable_abstains_never_ungrounded` |
| Retrieval empty | abstain / info-only; no supporting evidence | `test_retrieval_empty_abstains_with_no_supporting_evidence` |
| Baseline population-fallback | proceed but flag non-personalised; cap confidence | `test_population_fallback_proceeds_but_downgrades_and_caps_confidence` |
| De-identification uncertain | block external egress (default-deny) | `test_deidentification_uncertain_blocks_egress` |
| Schema / grounding violation | suppress/abstain, logged (persist + audit) | `test_grounding_violation_abstains_and_is_logged`, `test_scope_violation_is_suppressed_and_logged` |

## 2. NFRs

Run: `python -m scripts.loadtest`

### NFR-2 — ingestion ≥ 50 readings/s/instance — MEASURED, PASS
CPU-bound normalisation, so it is measurable on the Mac. The harness
(`measure_ingestion_throughput`) times `normalise_batch` over a synthetic batch;
observed rate is ~10^5 readings/s — orders of magnitude above the 50 readings/s
bar. Re-measure per deployment instance.

### NFR-1 — copilot p95 end-to-end ≤ 6 s — DEFERRED (`‹GPU-DEP›`)
Dominated by LLM inference on the H200. Per CLAUDE.md we do **not** load-test the
model on the Mac (single-tenant, ~20–50 tok/s, unrepresentative). The harness
(`measure_copilot_overhead`) measures only the deterministic retrieve+policy
overhead with the LLM stubbed — observed p95 well under 1 ms, i.e. essentially the
entire 6 s budget is available for inference. **Action on the H200:** run the same
path against the real vLLM backend and record end-to-end p95.

## 3. Production checklist (`08 §6`) — status

| Item | Status |
|---|---|
| `LLM_PROFILE=local`, `OPENROUTER_API_KEY` empty → no external egress | Code-enforced: production is hard-pinned to the `local` profile; the default-deny de-id gate guards every external profile (verified in §1). |
| Postgres + MinIO volumes on encrypted storage; TLS in front of api-gateway | **Ops-required** (infra, not code). |
| Secrets via Docker secrets / vault, not `.env` | **Ops-required.** |
| Network policy: only `llm-gateway` reaches host/external; only `api-gateway` exposed | Partially compose-expressed; enforce with the orchestrator's network policy. **Ops-required.** |
| Resource limits + healthchecks + restart policy on every service | **DONE (T7.3).** Every app service (state-engine, copilot, governance, ingestion, doc-coding) has a compose entry built from the shared `deploy/Dockerfile`, each with a `/healthz` healthcheck, CPU/memory limits, and `restart: unless-stopped`; only `api-gateway` is published. Verified on the Mac: both images build, all five service apps import in-container, and a running container serves `/healthz` + `/metrics`. Full multi-service `docker compose up` against the live data stack is the server bring-up step. |
| Prometheus `/metrics` + structured request logging, no PHI in logs | **DONE (T7.4).** `core.observability` gives every service `/metrics`, an `X-Trace-Id` on every response, and a structured JSON access log carrying only method / normalised path / status / duration / trace id (no body, query, or patient id). |
| Edge auth: JWT + RBAC + patient ownership, RFC-7807 errors | **DONE (T7.1).** The `api-gateway` authenticates every `/v1` request, RBAC- and ownership-gates it, and forwards to the owning service; errors are `application/problem+json`. |
| Global audit chain persists across restart, from every producer | **DONE (T7.2 + T7.2b).** `PERSISTENCE_BACKEND=postgres` config-switches the state-engine, **copilot, governance, and ingestion** to write the one hash-chained `audit_log` via a shared per-request transactional session (`core.db.request_session`). A restart test writes CONSENT_CHANGE + OUTCOME_CAPTURE + POLICY_DECISION + INGEST records, disposes the engine, re-opens, and confirms `verify_chain` still holds. |
| Queryable row tables: consent history, outcomes, outputs | **DONE (T7.2c).** The consent ledger writes the `consent` row (a grant is now visible to every `SqlConsentProvider`), the outcome store writes the `outcome` row (new table, migration `0003_outcome`), and the copilot writes the `output` row — all behind config-switched store ports. A restart test asserts every row survives and reads back. Qdrant vector persistence is config-switched into the copilot retriever (exercised once a KB corpus is loaded). |

## 4. Remaining infra (deferred to server bring-up)

- **NFR-1 end-to-end latency** against the real LLM on the H200 (`‹GPU-DEP›`).
- **TLS termination + secrets management** (Docker secrets / vault) and the
  orchestrator **network policy** — ops concerns, not code.
- Full multi-service `docker compose up` smoke against the live stack (image build +
  per-service startup are verified; the end-to-end run is a server-side check).

Persistence is no longer deferred: the Postgres-backed PSG, the **global audit chain
(from every producer)**, and the **consent / outcome / output row tables** are all
config-switchable and restart-verified on the Mac against real Postgres.
