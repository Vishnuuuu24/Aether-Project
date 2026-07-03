# 10 — Build Backlog (for the building agent)

Ordered. Each task lists its Definition of Done (DoD). Do not start a task until its predecessors' DoD pass. Implementation order is non-negotiable: **Schemas → APIs → Ingestion → State Engine → Event/Forecast → Retrieval → LLM → Policy/Tooling → UI client**. Build governance primitives (consent, audit, versioning) alongside, not last.

## Sprint 0 — Foundations
- **T0.1 Contracts.** Implement `schemas/` (reading, PSG nodes/edges/projection, output contract, FHIR + OMOP mapping). **DoD:** schemas import cleanly everywhere; round-trip serialise/validate tests pass; OpenAPI generates.
- **T0.2 Core libs.** `core/auth` (JWT+RBAC+consent), `core/audit` (hash-chained), `core/versioning`, `core/db` + alembic baseline. **DoD:** consent gate blocks uncovered access in tests; audit chain verifies; migrations apply.
- **T0.3 Infra up.** docker-compose: postgres, qdrant, redis, minio, api-gateway skeleton. **DoD:** `make up` healthy; `/readyz` green.

## Sprint 1 — Ingestion, quality, baseline, state
- **T1.1 Ingestion + adapters.** `/v1/ingest/readings`, replay + CSV adapters first; HealthKit/Health Connect/Fitbit adapters behind the same normaliser. **DoD:** readings missing required metadata are rejected with field errors; replay of PPG-DaLiA produces normalised readings.
- **T1.2 SQI + features.** `FeatureExtractor` (classical) + per-metric SQI thresholds. **DoD:** sub-threshold readings flagged `included_in_baseline=false`; features computed for required-core metrics.
- **T1.3 StatisticalBaselineEngine (05).** rolling median/MAD, EWMA, circadian stratification, sufficiency + population fallback. **DoD:** baselines personalise only after sufficiency; population fallback flagged; injected artefacts don't move the baseline (test).
- **T1.4 Patient State Engine + PSG.** versioned relational PSG, validate+commit, projection builder, consent scoping. **DoD:** state changes append-only + audited; `/state` returns a scoped projection with **no raw signals**.

## Sprint 2 — Event & forecast
- **T2.1 Event Engine (05 §6).** co-occurrence + persistence rules (versioned), severity. **DoD:** transient spikes suppressed; multi-metric events raised with contributing deviations; events not surfaced directly to patient.
- **T2.2 Forecast Engine (05 §7).** short-horizon per-metric forecasts + intervals behind `Forecaster`. **DoD:** forecasts produced for resting HR + sleep; MAE/RMSE harness runs.

## Sprint 3 — Documents & retrieval
- **T3.1 Doc-coding service (04 §4).** Docling/Marker OCR → MedCAT → FHIR (proposed); low-confidence stays `proposed`. **DoD:** discharge-summary sample yields coded Conditions/Medications/Observations; sub-threshold codes await confirmation.
- **T3.2 Hybrid retrieval.** BM25 + MedCPT/BGE dense in Qdrant + cross-encoder rerank over KB + patient record. **DoD:** Recall@K/MRR/nDCG harness runs on a seed corpus; retrieval is consent-scoped.

## Sprint 4 — LLM, policy, copilot
- **T4.1 LLM Gateway (06 §6).** profiles local/external_deidentified/dev; structured-output enforcement; de-id egress filter (default-deny). **DoD:** production pinned to local vLLM; external profile blocks PHI; structured schema enforced.
- **T4.2 Copilot orchestration (07 §5).** projection → retrieve → propose → policy → render. **DoD:** every claim carries an evidence ref; raw LLM output never returned.
- **T4.3 Policy Engine (06).** ordered deterministic checks, red-flag escalation, abstention, mandatory disclaimer/versions. **DoD:** ungrounded claims suppressed; scope violations blocked; red-flag forces seek-care + escalation; abstention returns reason; 100% outputs carry a policy decision record.

## Sprint 5 — Governance, eval, hardening
- **T5.1 Governance.** consent lifecycle, audit query API, outcome capture (`/v1/outcomes`), version registry. **DoD:** outer-loop outcomes recorded against prior outputs; audit reconstructs any output.
- **T5.2 Evaluation (11).** wire all harnesses; calibration; safety suite. **DoD:** all `11` metrics produce numbers on offline datasets; safety thresholds met or gaps logged.
- **T5.3 Hardening.** production checklist (`08 §6`), load test to NFR-2, latency to NFR-1 `‹GPU-DEP›`. **DoD:** NFRs measured; failure-mode defaults (`06 §9`) verified.

## Cross-cutting (every task)
Audit events emitted; versions stamped; tests + lint pass; no PHI in prompts/logs/external paths; interfaces respected (no new call sites for deferred components).

## Continuation
Sprints 0–5 build the engines. Wiring them into a running, exposed, deployable
system (integration, full API surface, edge/auth, real persistence, real-signal
eval) continues in `15_Post_v1_Backlog.md` (Sprint 6+).
