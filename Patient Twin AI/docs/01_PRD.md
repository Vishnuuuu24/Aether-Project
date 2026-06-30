# 01 — Product Requirements Document: Patient Copilot v1

## 1. Problem & thesis

Population reference ranges are blunt. A resting HR of 64 is "normal" for everyone and meaningful for no one. The product thesis is a **health twin**: a per-user model of *this individual's normal*, against which deviations are detected, explained, and (cautiously) acted on by a human. v1 delivers the patient-facing slice of that thesis.

## 2. Users & roles (v1)

| Role | Description | v1 access |
|---|---|---|
| Patient | Owns their data; receives explanations, flags, and guidance | Full read of own data; copilot Q&A; consent management |
| System (automated) | Ingests, computes baseline, detects deviations, drafts outputs | Internal |
| Clinician (read-only stub) | Receives escalations from red-flag outputs | Escalation queue only (full Doctor OS is out of scope) |

All patient identity is **pseudonymous** (UUID7). No direct identifiers in the analytic path.

## 3. Goals

- **G1.** Build a per-user baseline for each supported metric and detect deviations from it, context-stratified (resting/active/asleep), with calibrated confidence.
- **G2.** Answer patient questions about their own data, **grounded** in their Patient State Graph and a clinical knowledge base, with citations.
- **G3.** Ingest text clinical documents, code them, and fold them into the structured record so the copilot can reason over history.
- **G4.** Produce short-horizon forecasts on key personal metrics, as decision support only.
- **G5.** Guarantee that no output reaches a patient without passing the deterministic Policy Engine, and that every output is auditable.

## 4. Non-goals (v1)

- No diagnosis. No prescribing, dosing, or medication changes. No autonomous clinical action.
- No image/scan analysis (radiology vision). Documents are text only in v1.
- No Doctor/Hospital/Pharmacy product surfaces (the platform is shared, but only Patient Copilot ships in v1).
- No automated model retraining. The outer loop *captures* outcomes; retraining is human-gated and later.
- The copilot is **not** an emergency channel and must say so.

## 5. Functional requirements

### 5.1 Ingestion
- **FR-I1** Accept batched per-reading vitals via REST, normalised to the per-reading schema (`04 §2`). Every reading carries timestamp+tz, source device, quality flag, measurement context, and unit. Readings missing any required metadata field are rejected with a structured error.
- **FR-I2** Pluggable ingestion adapters: (a) HealthKit export, (b) Google Health Connect, (c) Fitbit Web API, (d) dataset-replay harness (PPG-DaLiA, WESAD, MESA, SHHS, WildPPG), (e) manual/CSV upload. All adapters emit the same normalised reading.
- **FR-I3** Accept document uploads (PDF/image-of-text/plain text). Run OCR → layout-aware text → clinical coding → FHIR resources (`04 §4`).
- **FR-I4** Enforce a valid, scoped, versioned consent record before any data is attached or processed.

### 5.2 Quality gating
- **FR-Q1** Every reading is scored by the Signal Quality Index (SQI). Readings below the configurable per-metric threshold are stored but **excluded from baseline learning** and flagged. ("Without this the model learns the noise.")

### 5.3 Baseline & deviation
- **FR-B1** Maintain a per-user, per-metric, per-context baseline (`05`). Cold-start falls back to age/sex population reference ranges, **explicitly labelled** as not-yet-personalised.
- **FR-B2** Score each new reading for deviation (magnitude, direction, confidence) against the relevant baseline.
- **FR-B3** Combine multi-metric deviations into candidate **events** via the Event Engine, with severity.

### 5.4 State
- **FR-S1** Maintain the Patient State Graph (`02 §4`, `04 §3`) as the versioned source of truth. Every state change is append-only and audited.

### 5.5 Retrieval & copilot
- **FR-R1** Hybrid retrieval (BM25 + dense MedCPT/BGE + cross-encoder rerank) over the clinical KB and the patient's own structured record.
- **FR-C1** The copilot answers a patient query by: reading PSG state (never raw signals) → retrieving evidence → drafting a structured output → **Policy Engine validation** → patient-facing rendering. Every claim cites PSG facts and/or KB passages.
- **FR-C2** If grounding is insufficient or confidence is below threshold, the copilot **abstains** with a clear reason and an escalation path.

### 5.6 Forecasting
- **FR-F1** Produce short-horizon (configurable, default 7-day) forecasts for resting HR, sleep duration, and other supported metrics, with prediction intervals. Surfaced as decision support, never as prediction of disease.

### 5.7 Guardrails & audit
- **FR-G1** The deterministic Policy Engine (`06`) runs on every output: allergy/interaction checks, red-flag escalation, confidence/abstention, scope limits. It can override the LLM.
- **FR-G2** Immutable audit log of every ingestion, state change, retrieval, LLM call, policy decision, and output.

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Copilot p95 end-to-end latency ≤ 6 s for a grounded answer at default context (`‹GPU-DEP›`). |
| NFR-2 | Ingestion sustains ≥ 50 readings/s/instance without backpressure loss. |
| NFR-3 | All PHI encrypted in transit (TLS) and at rest (DB + volume encryption). |
| NFR-4 | No PHI leaves the trust boundary. External model routing (OpenRouter) is dev-only and de-identified; production inference is self-hosted. |
| NFR-5 | RBAC + scoped consent enforced on every data access. |
| NFR-6 | Every model and ruleset is versioned; outputs record the versions that produced them. |
| NFR-7 | Deterministic components are reproducible given the same inputs and versions. |

## 7. Success criteria (v1 acceptance)

- Deviation flags evaluated against labelled events on offline datasets meet the targets in `11`.
- Copilot grounding: ≥ 95% of factual claims cite a PSG fact or KB passage; measured hallucination rate below target (`11`).
- Abstention correctness: the copilot abstains rather than guesses on out-of-grounding queries in ≥ 95% of adversarial eval cases.
- 100% of patient-facing outputs carry a Policy Engine decision record.
- Full audit trail reconstructable for any output.

## 8. Roadmap context

v1 = Phases 0–3 of the platform roadmap (Foundations → Patient State Engine → Knowledge/RAG → Patient Copilot), with Governance (Phase 4) primitives (consent, audit, policy) built in from the start rather than bolted on. Doctor/Hospital/Pharmacy (Phases 5–6) consume the same PSG and contracts later.

## 9. Regulatory posture (write it down, don't imply it)

v1 is **clinical decision support that proposes and explains; a human decides**. It is **not** a diagnostic device. This intended-use statement is load-bearing: it is what keeps the system in the lower-risk CDS lane. Implications: persistent not-a-doctor disclaimer, mandatory escalation path for red flags, no diagnostic/prescriptive output, human-in-the-loop for anything high-risk. Confirm the applicable regime for the launch geography (India: CDSCO medical-device rules + DPDP Act for data; abroad: HIPAA/GDPR). *This is a build constraint, not legal advice — get it reviewed by counsel.*
