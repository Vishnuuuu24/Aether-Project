# Patient Copilot / Twin Engine — v1 Build Handoff

**Status:** Build-ready specification for an autonomous building agent.
**Scope:** Patient Copilot v1 (the patient-facing slice of the shared Healthcare AI Platform).
**Audience:** The implementing agent + a human reviewer. Read `00` → `11` in order once, then treat each as a reference.

---

## 1. What this is

This package is the complete, self-contained specification needed to build the **Patient Copilot / Twin Engine v1**. It is intentionally prescriptive: schemas, contracts, model slugs, container topology, and per-task Definitions of Done are all pinned. Where a decision is deferred, it is marked `DEFERRED` with the interface that must remain stable so it can be swapped in later without a rewrite.

The non-negotiable architectural spine (do not refactor away):

1. **The deterministic engine is the primary intelligence. The LLM is an explanation/orchestration layer only.** The LLM never makes a clinical decision and never sees raw physiological signals.
2. **The Patient State Graph (PSG) is the single source of truth.** Nothing bypasses it. The LLM reads structured state from the PSG, never raw streams.
3. **The personal baseline is the product.** Deviation is measured against *this user's normal*, not population norms. A pipeline that only emits population-level classifications has missed the point of the system.
4. **Every output passes the deterministic Policy Engine before reaching a user.** The Policy Engine can override, downgrade, or suppress any LLM output.
5. **Closed-loop self-modification is forbidden.** The LLM never rewrites rulesets, schemas, or routing. Model/ruleset changes happen only through the human-gated outer loop with versioned releases.

## 2. v1 scope (locked)

| In scope | Out of scope (v1) |
|---|---|
| Vitals/wearable twin + personal baseline & deviation engine | Image / scan analysis (radiology vision) |
| Conversational copilot — grounded Q&A on the patient's own data | Doctor OS / Hospital OS / Pharmacy surfaces |
| Document ingestion: text documents (SOPs, clinical notes, discharge summaries, medical texts) → OCR → clinical coding → FHIR | Patient mobile/web UI (treated as a thin client of the output contract) |
| Short-horizon forecasting on personal baselines | Autonomous clinical actions of any kind |
| Deterministic guardrails + structured output contract | Closed-loop retraining (outer loop is captured but runs human-gated, later) |

## 3. Decisions locked for v1

- **Primary LLM:** Qwen3.6 35B A3B, **self-hosted via vLLM from day one**, behind an LLM Gateway abstraction. OpenRouter is a config-flip fallback for dev only — never in the PHI path in production.
- **No vision model resident** (image analysis deferred). Documents are handled as *text* via OCR.
- **Baseline engine = statistical-first** (robust rolling stats, EWMA, circadian-stratified, robust z-scores). The biosignal foundation encoder (PaPaGei-S / Pulse-PPG) is `DEFERRED` behind the `BaselineEngine` interface.
- **Data sources:** both real device APIs (HealthKit / Health Connect / Fitbit) and a dataset-replay + manual/CSV harness.
- **Storage:** PostgreSQL (FHIR + OMOP mirror; PSG realised as versioned relational tables) + Qdrant (vectors). No graph DB in v1.
- **Containerised:** all stateless services + Postgres + Qdrant via Docker Compose. vLLM runs on the host GPU; services route to it. An optional Compose profile runs vLLM in a container for single-host GPU setups.

## 4. Document map

| # | File | Purpose |
|---|---|---|
| 01 | `01_PRD.md` | Product requirements, users, success criteria, out-of-scope |
| 02 | `02_Architecture_Specification.md` | Module boundaries, data flow, sequence diagrams, **PSG definition** |
| 03 | `03_Tech_Stack_and_Model_Registry.md` | Pinned stack, model slugs, vLLM serving, VRAM budget |
| 04 | `04_Data_Contracts_and_Schemas.md` | FHIR mapping, per-reading schema, PSG schema, output contract |
| 05 | `05_Baseline_and_Deviation_Engine.md` | The differentiator — statistical baseline + deviation scoring |
| 06 | `06_Guardrails_Safety_Policy.md` | Deterministic Policy Engine, abstention, PHI/egress, audit |
| 07 | `07_API_Specification.md` | FastAPI endpoints / OpenAPI contract |
| 08 | `08_Docker_Stack.md` | docker-compose, services, vLLM routing, env, volumes |
| 09 | `09_Repository_Blueprint.md` | Monorepo layout |
| 10 | `10_Build_Backlog.md` | Ordered build tasks with per-task Definition of Done |
| 11 | `11_Evaluation_Plan.md` | Benchmarks + validation ladder |

## 5. Serving platforms (resolved)

Two confirmed environments, both behind the same OpenAI-compatible LLM Gateway (switching is a config change, not a code change):

- **Production — NVIDIA H200 DGX slice (80–100 GB):** vLLM on CUDA, fp8 preferred (4-bit AWQ fallback), real concurrency, container-GPU profile valid. PHI-allowed inside the trust boundary. See `03 §4a`, `08 §2a/§4`.
- **Local dev / demo — MacBook M5 Pro 48 GB:** mlx-lm on the macOS host (Metal cannot enter containers), 4-bit, single-tenant (~20–50 tok/s). Use `gpt-oss-20b`/a dense model as a known-good fallback while confirming the MoE quant. See `03 §4a`, `08 §2b`.

Remaining `‹GPU-DEP›` knobs (quantization, `MAX_MODEL_LEN`, concurrency) are now set per platform in `03 §4a` and `08 §2`. The Mac is a build/demo box; do not treat it as a multi-user server.

## 6. Definition of Done (whole v1)

The Patient Twin can: ingest device + document data, gate it on quality, maintain a per-user versioned Patient State Graph anchored on a **personal baseline**, detect and score **deviations from that baseline**, retrieve grounding evidence, answer grounded questions, produce short-horizon forecasts, and emit **structured outputs with confidence that have passed the deterministic Policy Engine** — with a full immutable audit trail and the outer-loop outcome capture wired (even though retraining itself is later).
