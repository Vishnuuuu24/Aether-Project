# 02 — Architecture Specification

## 1. Principles (freeze)

1. Deterministic Patient State Engine is the primary intelligence; the LLM is explanation/orchestration only.
2. The LLM never consumes raw physiological signals — only structured features and PSG state.
3. The Patient State Graph (PSG) is the single source of truth; nothing bypasses it.
4. The deterministic Policy Engine is the last gate before any user-facing output and can override the LLM.
5. No closed-loop self-modification: the LLM never edits rulesets, schemas, or routing.

## 2. Module boundaries

| Module | Responsibility | May NOT do |
|---|---|---|
| Ingestion Service | Adapter normalisation → per-reading schema; consent check | Compute baselines; call the LLM |
| SQI / Feature Extraction | Signal quality scoring + classical physiological features | Persist to PSG directly |
| Baseline & Deviation Engine | Per-user baselines; deviation scoring | Make clinical decisions; emit user text |
| Event Engine | Combine deviations → candidate events with severity | Escalate to a user directly (goes through Policy) |
| Patient State Engine (owns PSG) | Validate + commit state; versioning; source of truth | Generate explanations |
| Forecast Engine | Short-horizon forecasts on baselines | Predict disease; bypass Policy |
| Document/Coding Service | OCR → coding (LOINC/SNOMED/RxNorm) → FHIR | Decide clinical meaning |
| Retrieval Service | Hybrid search + rerank over KB + patient record | Generate answers |
| LLM Gateway | Single choke point for all model calls; local vLLM ↔ OpenRouter behind one interface | Persist clinical state; bypass Policy |
| Policy Engine | Deterministic validation/override of all outputs | Be implemented inside the LLM |
| Governance Service | Consent, RBAC, audit, model/ruleset version registry | — |
| API Gateway | AuthN/Z, routing, rate limiting | Hold business logic |

**Hard rule:** raw signals flow Ingestion → SQI/Features and are reduced to features there. Only features and structured state cross into the Patient State Engine and beyond. The LLM Gateway is physically downstream of the PSG and can only be handed structured context.

## 3. End-to-end data flow

```mermaid
flowchart TD
  subgraph DEV["Device / data sources"]
    A1[HealthKit / Health Connect / Fitbit]
    A2[Dataset replay harness]
    A3[Manual / CSV upload]
    A4[Document upload: PDF / text]
  end

  A1 --> ING[Ingestion Service<br/>normalise + consent check]
  A2 --> ING
  A3 --> ING
  A4 --> DOC[Document & Coding Service<br/>OCR -> code -> FHIR]

  ING --> SQI[SQI + Feature Extraction]
  SQI -->|features only, quality-gated| BASE[Baseline & Deviation Engine]
  BASE --> EVT[Event Engine]
  DOC --> PSE
  EVT --> PSE[Patient State Engine<br/>owns the PSG]
  BASE --> PSE
  FC[Forecast Engine] --> PSE
  PSE --> FC

  PSE -->|structured state, never raw signals| GW[LLM Gateway]
  RET[Retrieval Service<br/>BM25 + dense + rerank] --> GW
  KB[(Clinical KB)] --> RET
  PSE --> RET

  GW -->|proposed output| POL[Policy Engine<br/>deterministic, can override]
  POL -->|approved output contract| OUT[Output Contract]
  OUT --> APP[Patient app - thin client]
  POL -->|red flag| ESC[Clinician escalation queue]

  OUT -.outcome.-> GOV[Governance: audit + outcome capture]
  POL -.decision.-> GOV
  PSE -.version.-> GOV
```

## 4. The Patient State Graph (PSG) — definition

The PSG was previously undefined; this section is the fix. It is a **per-user, versioned, append-only typed graph** that is the source of truth for the patient's physiological and clinical state. It is **realised over PostgreSQL** (no graph DB in v1): nodes and edges are versioned relational tables exposed through a typed graph API. The LLM reads a *projection* of the PSG; it never writes to it.

### 4.1 Node types

| Node | Key fields |
|---|---|
| `Patient` | uuid7, age/dob, sex_at_birth, height, weight(+date), blood_group, disability? |
| `Metric` | metric_code, context, current_baseline_ref, latest_value, latest_ts, unit |
| `Baseline` | metric_code, context, center, dispersion, method, sample_n, window, confidence, computed_at, version |
| `Reading` | value, ts_tz, device, sqi, context, unit, included_in_baseline(bool) |
| `Deviation` | metric_code, magnitude, direction, z_robust, confidence, baseline_ref, ts |
| `Event` | type, severity, contributing_deviations[], onset_ts, status |
| `Condition` | snomed_code, status, onset, source_doc_ref |
| `Medication` | rxnorm_code, dose?, status, source_doc_ref |
| `Allergy` | substance_code, reaction, severity, source |
| `Observation` | loinc_code, value, unit, ts, source_doc_ref (from documents/labs) |
| `Document` | type (SOP/clinical-note/discharge/medical-text), uri, ocr_ref, codes[] |
| `Forecast` | metric_code, horizon, point[], interval[], method, generated_at |

### 4.2 Edge types

`has_reading`, `has_baseline`, `deviates_from` (Reading→Baseline), `aggregates` (Event→Deviation), `indicates` (Event→Condition, advisory), `contraindicates` (Medication↔Allergy/Condition), `derived_from` (Observation→Document), `forecasts` (Forecast→Metric), `supersedes` (versioning on any node).

### 4.3 Versioning & audit

Every node is immutable once written; updates create a new version linked by `supersedes`. The "current" view is a query over latest non-superseded versions. This gives free auditability and lets the outer loop reconstruct exactly what state produced any output.

### 4.4 What the LLM sees

A **PSG projection**: a compact, typed JSON snapshot (current baselines, recent deviations, active conditions/meds/allergies, relevant observations, latest forecast) — assembled by the Patient State Engine, scoped by consent, and stripped of raw signal arrays. See `04 §5`.

## 5. Copilot request sequence

```mermaid
sequenceDiagram
  participant U as Patient app
  participant API as API Gateway
  participant PSE as Patient State Engine
  participant RET as Retrieval Service
  participant GW as LLM Gateway (vLLM)
  participant POL as Policy Engine
  participant GOV as Governance/Audit

  U->>API: POST /copilot/query (auth, consent scope)
  API->>PSE: get PSG projection (consent-scoped)
  PSE-->>API: structured state (no raw signals)
  API->>RET: hybrid retrieve (query + state)
  RET-->>API: ranked evidence + citations
  API->>GW: prompt(state, evidence, query, output schema)
  GW-->>API: proposed output contract (info|flag|guidance)
  API->>POL: validate(proposed, PSG, rules)
  POL-->>API: approved | downgraded | suppressed | abstain (+reasons)
  API->>GOV: log(everything + versions)
  API-->>U: rendered patient-facing output (post-policy only)
```

## 6. Stable interfaces (so deferred parts swap cleanly)

```text
BaselineEngine:
  update(reading: Reading) -> None
  score(reading: Reading) -> DeviationResult
  get_baseline(metric, context) -> Baseline
# v1 impl: StatisticalBaselineEngine. DEFERRED impl: FoundationEncoderBaselineEngine.

FeatureExtractor:
  extract(window: SignalWindow) -> FeatureSet
# v1 impl: ClassicalFeatureExtractor (+SQI). DEFERRED: biosignal foundation encoder (PaPaGei-S/Pulse-PPG).

LLMGateway:
  complete(messages, schema, model_profile) -> StructuredOutput
# routes to local vLLM or OpenRouter by profile; identical call site.

Retriever:
  search(query, scope) -> list[EvidenceChunk]
```

Each deferred swap is a new implementation of an existing interface — no call-site changes, no schema changes.

## 7. Deployment topology

- Host machine: GPU + vLLM serving the primary model (`‹GPU-DEP›`).
- Docker network: all stateless services + Postgres + Qdrant. The LLM Gateway reaches vLLM via the host endpoint (default) or an in-Compose vLLM container (optional profile). See `08`.
- Trust boundary = the host/VPC. OpenRouter sits outside it and is reachable only from the LLM Gateway in `dev`/de-identified profiles.
