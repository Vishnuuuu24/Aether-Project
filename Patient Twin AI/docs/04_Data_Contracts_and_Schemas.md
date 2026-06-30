# 04 — Data Contracts & Schemas

JSON Schema-style definitions. These are contracts: the building agent implements them exactly. All timestamps are RFC 3339 with timezone. All identifiers are UUID7 unless stated.

## 1. Patient profile (onboarding, once)

```json
{
  "patient_id": "uuid7",
  "consent": {
    "scope": ["vitals", "documents", "copilot", "forecast"],
    "version": "string",
    "granted_at": "rfc3339",
    "revoked_at": "rfc3339|null"
  },
  "age_or_dob": {"dob": "date|null", "age_years": "int|null"},
  "sex_at_birth": "male|female|intersex|unknown",
  "gender": "string|null",
  "height_cm": "number|null",
  "weight": {"value_kg": "number", "measured_at": "rfc3339"},
  "blood_group": "string|null",
  "physical_disability": "string|null"
}
```
**Rules:** no processing without a valid, non-revoked consent covering the relevant scope. `age_or_dob` and `sex_at_birth` are load-bearing for population fallback ranges and must be present before baseline cold-start fallback can run.

## 2. Per-reading schema (every single data point)

```json
{
  "reading_id": "uuid7",
  "patient_id": "uuid7",
  "metric_code": "string",          // canonical metric code (see §2.1)
  "value": "number | object",       // scalar, or structured (e.g. sleep stages)
  "unit": "string",                 // explicit; never assumed
  "timestamp": "rfc3339",           // with timezone; readings without tz are rejected
  "source_device": "string",        // e.g. apple_watch_s9, fitbit_charge6, dataset:PPG-DaLiA
  "sqi": "number",                  // 0..1 signal quality / confidence
  "context": "resting|active|asleep|post_meal|unknown",
  "included_in_baseline": "bool",   // set by SQI gate, not by the sender
  "ingest_adapter": "string",
  "raw_ref": "uri|null"             // pointer to raw window in object store; NOT sent downstream of feature extraction
}
```
**Rejection rule:** missing any of `metric_code, value, unit, timestamp(tz), source_device, context` → 422 with field-level error. `sqi` absent → adapter must compute or mark `unknown` (treated as below threshold).

### 2.1 Supported metrics (v1)

| Tier | Metrics |
|---|---|
| Required core | heart_rate (resting + continuous), steps/activity |
| Strongly recommended | sleep (duration+stages), spo2, respiratory_rate, skin_temp_trend |
| Optional (use if present) | ecg, hrv (if beat-to-beat exposed), glucose/cgm, bp_cuff, vo2max, gait/steadiness, weight_trend, menstrual_cycle, smoking/alcohol, lab observations |

The engine must run on **required core alone**; everything else degrades gracefully and is `null`-tolerant.

## 3. Patient State Graph schema (relational realisation)

Core tables (abbreviated; all carry `id uuid7, patient_id, version int, supersedes uuid|null, created_at, created_by`):

```sql
metric_node(metric_code, context, unit, latest_value, latest_ts, current_baseline_id)
baseline_node(metric_code, context, method, center, dispersion, sample_n,
              window_spec, confidence, is_population_fallback bool, computed_at)
reading_node(metric_code, value jsonb, unit, ts_tz, source_device, sqi,
             context, included_in_baseline)
deviation_node(metric_code, baseline_id, magnitude, direction, z_robust,
               confidence, ts)
event_node(type, severity, status, onset_ts, contributing_deviation_ids uuid[])
condition_node(snomed_code, status, onset, source_document_id)
medication_node(rxnorm_code, dose, status, source_document_id)
allergy_node(substance_code, reaction, severity, source)
observation_node(loinc_code, value, unit, ts, source_document_id)
document_node(doc_type, uri, ocr_artifact_uri, code_ids uuid[])
forecast_node(metric_code, horizon, points jsonb, intervals jsonb, method, generated_at)
edge(from_id, from_type, to_id, to_type, edge_type, attrs jsonb)
```
**Immutability:** rows are never updated in place; a change writes a new version row and sets `supersedes`. The "current PSG" is `WHERE id NOT IN (SELECT supersedes ...)`.

## 4. Document → FHIR mapping

Pipeline: upload → Docling/Marker OCR → section/layout parse → MedCAT coding → FHIR resources committed as `proposed`, then validated → `committed`.

| Document content | FHIR resource | Codes |
|---|---|---|
| Lab result | `Observation` | LOINC |
| Diagnosis / problem | `Condition` | SNOMED CT |
| Medication | `MedicationStatement` | RxNorm |
| Allergy | `AllergyIntolerance` | SNOMED/RxNorm |
| Discharge summary / note | `DocumentReference` + extracted resources | mixed |

A persistent **OMOP mirror** is maintained alongside FHIR for analytics/eval. Coding confidence below threshold → resource stays `proposed` and is surfaced for human confirmation (never silently committed).

## 5. PSG projection (the only thing the LLM sees)

```json
{
  "patient": {"age_years": "int", "sex_at_birth": "string"},
  "baselines": [{"metric_code","context","center","dispersion","confidence","is_population_fallback"}],
  "recent_deviations": [{"metric_code","direction","magnitude","z_robust","confidence","ts"}],
  "active_events": [{"type","severity","onset_ts"}],
  "conditions": [{"snomed_code","display","status"}],
  "medications": [{"rxnorm_code","display","status"}],
  "allergies": [{"substance","reaction","severity"}],
  "recent_observations": [{"loinc_code","display","value","unit","ts"}],
  "latest_forecast": [{"metric_code","horizon","points","intervals"}],
  "as_of": "rfc3339",
  "consent_scope": ["..."],
  "versions": {"baseline_engine","ruleset","prompt","model"}
}
```
**No raw signal arrays. No `reading_node.raw_ref`.** Consent-scoped: fields outside granted scope are omitted.

## 6. Output contract (every user-facing output)

```json
{
  "output_id": "uuid7",
  "patient_id": "uuid7",
  "type": "info | flag | guidance",
  "message": "string",                 // patient-facing text, generated only after policy approval
  "severity": "none | low | moderate | high",
  "confidence": "number",              // 0..1, calibrated
  "evidence": [
    {"kind": "psg_fact|kb_passage", "ref": "string", "quote_or_fact": "string"}
  ],
  "baseline_reference": {"metric_code","center","dispersion","is_population_fallback"} ,
  "recommended_action": "none | monitor | lifestyle_info | seek_care | seek_urgent_care",
  "escalation": {"triggered": "bool", "reason": "string|null"},
  "abstained": {"value": "bool", "reason": "string|null"},
  "policy": {"decision": "approved|downgraded|suppressed|abstain", "rule_ids": ["..."]},
  "disclaimer": "string",              // mandatory not-a-doctor + not-an-emergency-channel
  "versions": {"model","ruleset","baseline_engine","prompt"},
  "created_at": "rfc3339"
}
```
**Constraints:** `recommended_action` is a closed vocabulary — no free-text actions, no diagnoses, no dosing. `message` must be consistent with `evidence`; any claim without an `evidence` ref is a contract violation and is suppressed by Policy. Every output, including abstentions and suppressions, is persisted and audited.

## 7. Audit record

```json
{
  "audit_id":"uuid7","patient_id":"uuid7","actor":"system|patient|clinician",
  "action":"ingest|sqi|baseline_update|state_commit|retrieve|llm_call|policy_decision|output|consent_change",
  "input_refs":["..."],"output_refs":["..."],"versions":{...},
  "timestamp":"rfc3339","prev_hash":"string","hash":"string"
}
```
Hash-chained (`prev_hash`) for tamper-evidence; append-only.
