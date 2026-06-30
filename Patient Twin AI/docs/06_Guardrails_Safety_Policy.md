# 06 — Guardrails, Safety & Policy Engine

The Policy Engine is **deterministic, separate from the LLM, and the last gate before any user-facing output.** It can approve, downgrade, suppress, or force-abstain on any LLM proposal. It is versioned and never edited by the LLM.

## 1. Position in the flow

```
PSG projection + evidence ──► LLM Gateway ──► proposed output contract ──► POLICY ENGINE ──► user
                                                                              │
                                                                              └► clinician escalation (red flags)
```
The LLM only ever *proposes*. The Policy Engine decides what ships.

## 2. Deterministic checks (run in order; first hard failure wins)

1. **Schema validity.** Output must match the `04 §6` contract exactly. Invalid → suppress + log.
2. **Grounding check.** Every factual claim in `message` must map to an `evidence` ref (PSG fact or KB passage). Ungrounded claim → suppress (or downgrade to abstain). This is the primary anti-hallucination gate and is mechanical, not model-judged.
3. **Scope check.** No diagnosis, no prescribing/dosing, no medication changes, no contradiction of a recorded clinician instruction. `recommended_action` must be in the closed vocabulary. Violation → suppress.
4. **Allergy / interaction check.** Cross-reference any mentioned substance against PSG `Allergy` and `Medication` nodes (RxNorm). Conflict → suppress the suggestion and raise a flag.
5. **Red-flag escalation.** Configured acute patterns (versioned ruleset) force `recommended_action ∈ {seek_care, seek_urgent_care}`, set `escalation.triggered=true`, and enqueue a clinician escalation — **regardless of LLM output**. Examples (illustrative, to be set with clinical input): sustained marked tachycardia at rest with corroborating deviations; SpO₂ sustained below configured floor; user-reported emergency symptoms in the query.
6. **Confidence threshold.** If `confidence < τ` (per output type), force abstain with reason.
7. **Population-fallback honesty.** If the supporting baseline is `is_population_fallback`, the message must not claim personalised normal; Policy rewrites/downgrades accordingly.

## 3. Abstention

Abstention is a first-class, *correct* outcome, not a failure. The copilot abstains when grounding is insufficient, confidence is low, or the query is outside scope (e.g. asking for a diagnosis). Abstention returns a clear reason and an escalation/next-step path. Abstention correctness is an explicit eval metric (`11`).

## 4. Mandatory output elements

- **Disclaimer** on every output: not a doctor; not an emergency service; for emergencies contact local emergency services. (Localised text.)
- **Escalation path** present whenever `severity >= moderate` or `escalation.triggered`.
- **Versions** stamped (model, ruleset, baseline engine, prompt).

## 5. Prohibited outputs (hard)

- Diagnoses stated as fact; prognoses of disease.
- Medication, dose, or treatment changes.
- Anything contradicting a recorded clinician instruction.
- Acting as the sole channel in an emergency.
- "Your normal is X" when X is a population fallback.

## 6. PHI & external routing

- **Production inference is self-hosted.** No PHI leaves the trust boundary.
- The LLM Gateway exposes profiles: `local` (vLLM, PHI-allowed), `external_deidentified` (OpenRouter, PHI-forbidden, payload must pass de-identification), `dev` (synthetic only). Production patient traffic is hard-pinned to `local`.
- A de-identification filter sits in front of any `external_*` profile and blocks egress if identifiers are detected. Default-deny.
- Document/object storage and DB are encrypted at rest; TLS in transit.

## 7. Human-in-the-loop

- Red-flag and high-severity outputs are queued for clinician review (the read-only escalation stub in v1). The patient still receives a safe, policy-approved "seek care" message immediately; the clinician path is for follow-up, not for blocking acute safety messaging.
- Newly `proposed` coded clinical resources below confidence threshold await human confirmation before becoming `committed` PSG facts.

## 8. Audit & governance

- Every policy decision (approve/downgrade/suppress/abstain) is logged with `rule_ids`, inputs, and versions, hash-chained per `04 §7`.
- The **outer loop** captures real outcomes (admission, diagnosis, medication changes) against the outputs that preceded them — stored for later human-gated retraining. v1 captures and stores; it does **not** retrain automatically.
- Ruleset and prompt changes are versioned releases reviewed by a human; the system cannot self-modify them.

## 9. Failure modes & defaults

| Failure | Default behaviour |
|---|---|
| LLM Gateway unavailable | Abstain with "temporarily unavailable"; never fall back to ungrounded generation |
| Retrieval empty | Abstain or info-only with explicit "no supporting evidence found" |
| Baseline population-fallback | Proceed but flag non-personalised; lower confidence |
| De-identification uncertain | Block external egress (default-deny) |
| Schema/grounding violation | Suppress, log, alert |
