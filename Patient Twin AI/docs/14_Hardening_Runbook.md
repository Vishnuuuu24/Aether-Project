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
| Resource limits + healthchecks + restart policy on every service | `healthz`/`readyz` exist on each service app; **compose still builds only `postgres`/`qdrant`/`redis`/`minio`/`api-gateway`.** Containerising the remaining app services (state-engine, ingestion, policy-engine, copilot, governance, …) with per-service Dockerfiles, resource limits and restart policies is the remaining infra step — deferred to the H200 bring-up alongside NFR-1. |

## 4. Remaining infra (deferred to server bring-up)

- Per-service Dockerfiles + compose entries (only `api-gateway` has one today).
- Real DB-backed wiring (services run in-memory dev wiring; production injects
  Postgres/Qdrant stores via DI — the seams already exist).
- NFR-1 end-to-end measurement against the real LLM on the H200.

None of these are Mac-measurable; they are recorded here rather than faked.
