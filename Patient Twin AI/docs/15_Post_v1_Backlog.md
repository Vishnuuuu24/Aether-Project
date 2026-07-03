# 15 — Post-v1 Build Backlog (Sprint 6+)

Continuation of `10_Build_Backlog.md`, which ends at Sprint 5. The Sprint 0–5
*engines* are built and unit-tested in isolation; the work below **wires them into
a running, exposed, deployable system** and turns evaluation from synthetic to
real — all on the Mac, no GPU or new licensed datasets required (those are the
separate deferred track in §D).

Every task here traces to an existing spec — nothing new is invented. Same rules
as `10 §Cross-cutting` apply to all tasks: audit events emitted, versions stamped,
tests + lint + mypy pass, no PHI in prompts/logs/external paths, deferred
components stay behind their interfaces (CLAUDE.md).

Ordering is by dependency: Sprint 6 (integration + surface) unblocks Sprint 7
(edge/runtime); Sprint 8 (real signal) can run in parallel once 6.1 lands.

---

## Sprint 6 — Integration & full API surface

Goal: prove the built engines work *together*, and expose the contract that
`07_API_Specification.md` already defines but that only 3 of ~20 endpoints
implement today (`/state`, `/ingest/readings`, `/copilot/query`, + governance).

### T6.1 End-to-end integration harness (validation ladder rung 2)
*Source: `11 §2` rung 2 ("Integration — full request path on synthetic patients;
audit reconstruction"); `02` data flow.*
- **DoD:** one test drives a synthetic patient through ingestion → SQI/features →
  baseline → PSG commit → event/forecast → copilot answer, asserting the final
  `OutputContract` is Policy-approved and that the audit chain reconstructs the
  full path (`verify_chain` passes; every mutation has a record).
- **Procedure:** (1) build a synthetic reading stream fixture; (2) compose the real
  engines with in-memory stores sharing ONE audit writer; (3) run the pipeline
  stage by stage; (4) assert contracts at each seam + final grounding; (5) assert
  audit reconstruction of the copilot output.
- **Do:** reuse the real engines and existing fixtures; share a single audit chain
  so reconstruction is meaningful. Keep it deterministic.
- **Don't:** mock an engine you're supposed to be integrating; don't introduce a
  parallel "integration-only" code path — the seams must be the production ones.

### T6.2 Twin/state read endpoints
*Source: `07 §4`.*
- **DoD:** `GET /v1/patients/{id}/baselines?metric=&context=`, `/deviations?since=`,
  `/events?status=`, `/forecast?metric=&horizon=`, `/observations?code=`,
  `/documents` all return consent-scoped projections with **no raw signals**;
  unknown patient → 404, no covering consent → 403.
- **Procedure:** (1) add read methods to the state-engine service over the PSG;
  (2) add endpoints mirroring the `/state` handler's consent/404/403 handling;
  (3) test each for scoping + the deny-by-default paths.
- **Do:** reuse the existing `PSGProjection`/consent-gate machinery; keep responses
  contract-typed.
- **Don't:** leak reading-level data or any field the projection intentionally
  drops (CLAUDE.md principle 2); don't add a new consent mechanism.

### T6.3 Clinician escalation endpoints (read-only stub, v1)
*Source: `07 §6`.*
- **DoD:** `GET /v1/escalations?status=open` returns queued red-flag/high-severity
  outputs; `POST /v1/escalations/{id}/ack` records a clinician ack and **audits
  it** (actor=clinician).
- **Procedure:** (1) back the `EscalationSink` with a queryable store; (2) expose
  the queue; (3) implement ack → audit record; (4) test enqueue-on-escalation →
  list → ack → audit.
- **Do:** reuse the copilot's existing escalation enqueue path; stamp versions.
- **Don't:** let ack mutate or re-open the underlying output; don't surface raw
  events to a patient-facing route.

### T6.4 Ingestion surface: documents, replay, webhooks
*Source: `07 §3`; `doc_coding_service` exists but has **no HTTP app**; replay is a
CLI only today.*
- **DoD:** `POST /v1/ingest/documents` (multipart → OCR→coding→FHIR *proposed*),
  `POST /v1/ingest/replay` (`{dataset, patient_id, speed}`, dev-only), and the
  adapter webhook route accept payloads and route through the existing normaliser /
  doc-coding pipeline; low-confidence codes stay `proposed`.
- **Procedure:** (1) add an `app/main.py` to `doc_coding_service`; (2) wrap the
  existing replay adapter behind the HTTP route; (3) wire multipart → doc-coding;
  (4) test rejection/validation and the `proposed` gate.
- **Do:** reuse `normalise_batch` and the doc-coding service; keep replay dev-only.
- **Don't:** auto-confirm sub-threshold codes; don't invent OCR/coding output.

### T6.5 Copilot streaming variant
*Source: `07 §5` ("Streaming variant `/copilot/query:stream` (SSE) streams only
**after** Policy approval of the structured plan").*
- **DoD:** SSE endpoint streams tokens only after the Policy Engine approves the
  structured output; abstained/suppressed cases stream nothing un-vetted.
- **Procedure:** (1) run the full decide() first; (2) only on approval, stream the
  approved message; (3) test that a suppressed/abstained decision yields no
  streamed content.
- **Do:** keep the deterministic gate strictly before any streaming.
- **Don't:** stream raw LLM tokens ahead of Policy approval (the whole point).

---

## Sprint 7 — Edge, auth & runtime (make it deployable on the Mac)

Goal: a real gateway, real persistence, and the full stack running under
`docker compose` locally (minus the GPU `vllm` profile).

### T7.1 api-gateway: auth, RBAC, consent-scope, tracing, error contract
*Source: `07 §1` & `07 §9`; primitives already in `core/auth/` (jwt, rbac,
consent_gate). Gateway today is health-only (`asyncpg`/`qdrant` readiness).*
- **DoD:** every `/v1` route requires a valid JWT; access is RBAC- and
  consent-scoped; each response carries `X-Trace-Id`; errors use RFC-7807
  problem+json; unauthenticated/forbidden paths tested.
- **Procedure:** (1) auth middleware validating JWT → principal; (2) RBAC + consent
  dependency; (3) trace-id middleware; (4) RFC-7807 exception handlers; (5) route
  to services (proxy or mounted routers).
- **Do:** enforce server-side; wire the existing `core/auth` primitives.
- **Don't:** trust client-supplied scopes; don't let any route bypass the Policy
  Engine or the consent gate (`07 §10`).

### T7.2 Persistence wiring (DB-backed stores active)
*Source: `08`; `T0.2/T0.3`; SQL stores exist (`sql_store.py`) but app mains use
in-memory dev wiring.*
- **DoD:** services run against Postgres (PSG, audit, outcomes, consent) and Qdrant
  (vectors) via DI; alembic migrations apply cleanly; the audit chain persists and
  `verify_chain` holds across a restart.
- **Procedure:** (1) activate the SqlAlchemy stores behind the existing store
  interfaces; (2) finalize/extend alembic migrations; (3) config-switch in/out via
  env; (4) integration test on a real (compose) DB.
- **Do:** keep the store interface the seam (no engine change); one transaction per
  request where the spec requires it.
- **Don't:** fork a "DB version" of any service — it's config, not code (CLAUDE.md).

### T7.3 Full-stack containers
*Source: `08 §6`, `08 §3` (compose declares services that don't yet have
Dockerfiles); `14 §3–4` (documented gap).*
- **DoD:** every app service has a Dockerfile + compose entry with healthcheck,
  resource limits, and restart policy; `docker compose up` brings up the full stack
  on the Mac (GPU `vllm` profile excluded); only `api-gateway` is exposed.
- **Procedure:** (1) per-service Dockerfile (reuse the api-gateway one as template);
  (2) compose entries with limits/healthchecks; (3) verify `/readyz` green across
  the stack; (4) update `14`.
- **Do:** mirror the existing api-gateway Dockerfile/compose conventions.
- **Don't:** enable the `gpu`/`vllm` profile on macOS (Metal can't enter Docker —
  CLAUDE.md); point `LLM_GATEWAY_BASE_URL` at host LM Studio instead.

### T7.4 Observability
*Source: `07 §8` (`GET /metrics` Prometheus); no `/metrics` exists today.*
- **DoD:** each service exposes Prometheus `/metrics`; structured request logging
  with `X-Trace-Id`; **no PHI in any log line**.
- **Procedure:** (1) add a metrics exporter; (2) structured logging middleware;
  (3) a log-scrub/assert test for PHI-shaped content.
- **Do:** counts and IDs only in logs.
- **Don't:** log prompts, messages, or reading values.

---

## Sprint 8 — Real signal & real numbers (turn eval synthetic → real, no GPU)

Goal: close the T5.2 gaps that are CPU-doable, so the eval report produces numbers
on real offline data instead of synthetic smoke.

### T8.1 Classical HR/HRV-from-waveform FeatureExtractor
*Source: CLAUDE.md ("classical features in v1; PaPaGei-S/Pulse-PPG deferred behind
`FeatureExtractor`"); `03`; `05`. Today `ClassicalFeatureExtractor` only does
descriptive stats over already-scalar readings — it cannot derive HR from raw
ECG/PPG.*
- **DoD:** a classical DSP path derives heart-rate (and where feasible HRV) from raw
  ECG/PPG windows, behind the existing `FeatureExtractor` interface; validated
  against a known-answer signal.
- **Procedure:** (1) implement peak detection (e.g. Pan–Tompkins-style for ECG,
  systolic-peak for PPG) → beat intervals → HR; (2) plug behind `FeatureExtractor`;
  (3) test on a synthetic waveform with known HR; (4) sanity-check on one WESAD
  subject.
- **Do:** keep it classical/CPU; behind the interface (no new call sites).
- **Don't:** introduce a DL encoder (that's the deferred, server-side job); don't
  invent physiological thresholds.

### T8.2 Offline-dataset deviation-eval adapters → real §1.2 numbers
*Source: `11 §1.2`, `01` FR-I2; datasets already on disk (WESAD, PPG-DaLiA).*
- **DoD:** an adapter parses WESAD (and/or PPG-DaLiA) into labelled
  `LabelledDeviation`s (using T8.1 for HR), and `ai/eval_report.py` produces real
  precision/recall/F1 + ECE on that data, replacing the synthetic section; the
  WESAD `DATASET` gap in the report is closed and the layout is schema-validated.
- **Procedure:** (1) validate the WESAD `.pkl` layout + label alignment; (2) map
  stress/baseline windows → labels; (3) feed the deviation-eval harness; (4) wire
  the section into the report with `dataset="WESAD"`.
- **Do:** validate the layout before trusting labels (see `datasets/WESAD/README`).
- **Don't:** fabricate a benchmark from signals you haven't verified.

### T8.3 Clinical-config loaders & validation *(content is clinician-gated)*
*Source: `05`, `06`; config stubs are UNSET by design.*
- **DoD:** loaders + schema validation for red-flag patterns, SQI thresholds,
  confidence thresholds, prohibited-term lexicon, and KB content — with clear
  failure when malformed; safe/inert when UNSET (as today).
- **Procedure:** (1) define the config schema per `05/06`; (2) validating loader;
  (3) tests for malformed/empty; (4) leave content as labelled stubs.
- **Do:** build the *structure*; keep the *content* a clinician deliverable.
- **Don't:** invent thresholds, dosages, red-flag patterns, or guideline text
  (CLAUDE.md — hard rule). Fabricated clinical content is a defect, not a stub.

---

## D. Deferred track (NOT tasked here — external blockers)

Recorded so the roadmap is honest about what's *out* of the Mac/no-dataset scope:

- **Server/GPU (H200):** NFR-1 end-to-end latency measurement; production vLLM;
  Qwen3.6 35B QLoRA fine-tune (`CLAUDE.md`, `12`, `14 §2`).
- **Licensed/credentialed datasets:** MIMIC-IV Notes, UMLS, RxNorm, SNOMED, LOINC
  (`13`) — human-required accounts/DUAs.
- **DL biosignal encoders:** PaPaGei-S / Pulse-PPG behind `FeatureExtractor`
  (deferred by design; server training).
- **UI thin client:** the patient app (`10` build order ends at "UI client";
  CLAUDE.md — thin client). Separate frontend track, out of this backend backlog.
- **Clinician-provided content:** the actual red-flag/SQI/KB values feeding T8.3.
