# 11 — Evaluation & Validation Plan

Two layers: **component benchmarks** (does each module work) and the **validation ladder** (is the whole thing safe to put in front of people). Targets below are starting bars to refine with clinical input; the harnesses are mandatory regardless of the numbers.

## 1. Component benchmarks

### 1.1 Retrieval
- Recall@K, MRR, nDCG over a labelled clinical QA set on the seed KB.
- Reranker lift: top-k accuracy with vs without the cross-encoder.
- **Bar (starting):** rerank improves nDCG@10 over dense-only by a meaningful margin; Recall@20 ≥ 0.9 on the seed set.

### 1.2 Baseline & deviation (the differentiator)
- **Deviation detection:** precision/recall/F1 of flags against labelled events on offline datasets (WESAD stress, MESA/SHHS sleep events, PPG-DaLiA activity context, WildPPG noise robustness).
- **Calibration:** reliability curve + Expected Calibration Error on confidence. A 0.7 confidence should be correct ~70% of the time. **Bar:** ECE ≤ 0.1.
- **Robustness:** inject artefacts that fail SQI → baseline center/dispersion must not move beyond tolerance.
- **Fallback honesty:** no `personalised` label before sufficiency; correct transition logging.

### 1.3 Forecast
- MAE / RMSE per metric vs naive (last-value, seasonal-naive) baselines. **Bar:** beat seasonal-naive on resting HR and sleep duration.
- Interval calibration: empirical coverage of prediction intervals matches nominal.

### 1.4 LLM / copilot safety (gating)
- **Grounding rate:** % of factual claims with a valid evidence ref. **Bar:** ≥ 0.95.
- **Hallucination rate:** % outputs with an unsupported claim that slipped past Policy. **Bar:** → 0 (any non-zero is a defect, since grounding is a mechanical gate).
- **Abstention correctness:** on an adversarial out-of-grounding / out-of-scope set (diagnosis requests, emergencies, ungrounded asks), the system abstains/escalates. **Bar:** ≥ 0.95.
- **Scope-violation rate:** diagnoses/dosing/prescribing in output. **Bar:** 0 (hard).
- **Red-flag recall:** configured acute patterns always trigger escalation. **Bar:** 1.0 on the red-flag test set.
- **Policy coverage:** 100% of outputs carry a decision record.

### 1.5 Tooling / orchestration
- Tool/call success rate; end-to-end latency (NFR-1 `‹GPU-DEP›`); ingestion throughput (NFR-2).

## 2. Validation ladder (do not skip rungs)

1. **Unit** — per-module logic, contracts, deterministic checks.
2. **Integration** — full request path on synthetic patients; audit reconstruction.
3. **Offline clinical** — run against the offline datasets; produce all §1 metrics.
4. **Clinician review** — clinicians review a sample of outputs for safety/appropriateness; red-flag and abstention behaviour audited by a human.
5. **Silent deployment** — system runs on real data producing outputs **not shown to patients**; compare against outcomes/clinician judgement.
6. **Prospective monitoring** — limited live exposure with monitoring + kill switch.
7. **Drift detection + recalibration** — monitor input/feature/score drift; recalibrate on schedule; any model/ruleset change is a **human-gated versioned release** (no self-modification).

## 3. Outer-loop capture (built in v1, retrain later)

Record real clinical outcomes (admission, diagnosis, medication change) via `/v1/outcomes`, linked to the outputs and PSG versions that preceded them. This is the labelled signal for later human-gated retraining of adapters / task heads / reranker. v1 captures and stores only.

## 4. Reporting

Every eval run is versioned (model, ruleset, prompt, schema, dataset) and stored. Dashboards track the §1 metrics over time; safety metrics (grounding, scope, red-flag, abstention) are release gates — a regression blocks deploy.
