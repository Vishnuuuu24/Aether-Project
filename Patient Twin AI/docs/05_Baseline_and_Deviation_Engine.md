# 05 — Personal Baseline & Deviation Engine

This is the differentiator. It is what makes the system a *twin* rather than a generic classifier. v1 is **statistical-first** and fully specified here; the foundation-encoder version is `DEFERRED` behind the same interface.

## 1. Objective

For each patient, each metric, and each context bucket, maintain a robust estimate of *this user's normal*, and score new readings as deviations from it with calibrated confidence. Population ranges are only a cold-start fallback, always labelled as such.

## 2. Interface (stable)

```python
class BaselineEngine(Protocol):
    def update(self, reading: Reading) -> None: ...
    def score(self, reading: Reading) -> DeviationResult: ...
    def get_baseline(self, metric_code: str, context: str) -> Baseline: ...

# v1: StatisticalBaselineEngine
# DEFERRED: FoundationEncoderBaselineEngine (PaPaGei-S / Pulse-PPG embeddings -> per-user density)
```

`DeviationResult = {metric_code, magnitude, direction(+/-), z_robust, confidence, baseline_ref, is_population_fallback}`.

## 3. Quality gate (precondition)

Only readings with `sqi >= threshold[metric]` update the baseline (`included_in_baseline=true`). Sub-threshold readings are stored, scored with reduced confidence, and **never pollute the baseline**. Thresholds are per-metric config. This directly implements the data-list principle: *without the quality flag the model learns the noise.*

## 4. Baselines (v1 statistical method)

Per `(patient, metric_code, context)`:

- **Center:** rolling **median** over the trailing window (default 28 days, configurable per metric). Robust to outliers.
- **Dispersion:** **MAD** (median absolute deviation), scaled to a robust σ (`1.4826 × MAD`).
- **Recent trend:** **EWMA** (half-life configurable, default 7 days) tracked alongside the long window to catch drift.
- **Circadian / seasonal stratification:** where data density allows, condition the baseline on time-of-day bucket (and day-of-week for activity-linked metrics). Sleep, resting HR, and temperature are strongly circadian — stratify these by default.
- **Sample sufficiency:** a baseline is `personalised` only when `sample_n >= min_n[metric]` (default 50 quality-passing readings in-bucket) **and** spans `>= min_days` (default 7). Otherwise `is_population_fallback=true`.

### 4.1 Cold start / fallback

Until sufficiency is met, use age/sex population reference ranges (from the profile) as the baseline, with `is_population_fallback=true`. Every downstream artefact (deviation, output) inherits and surfaces this flag so the patient is never told "your normal" when it's really "the population's normal."

## 5. Deviation scoring

For a reading `x` against baseline `(center c, robust σ)`:

- **Modified z-score:** `z_robust = (x − c) / σ_robust`. (For non-Gaussian metrics, use the empirical quantile of `x` in the trailing in-bucket distribution and convert to a z-equivalent.)
- **Direction:** sign of `(x − c)`.
- **Magnitude:** `|z_robust|`, bucketed: `<2` normal, `2–3` mild, `3–4.5` moderate, `>4.5` marked.
- **Confidence:** function of `sqi`, `sample_n`, dispersion stability, and whether population fallback is in use. Low `sample_n` or low `sqi` → low confidence even for large `z`. Confidence is **calibrated** against held-out data (`11`); do not ship raw heuristic confidences as if calibrated.

## 6. Event Engine (multi-metric combination)

Single-metric deviations are weak signals. The Event Engine combines them into candidate **events**:

- **Co-occurrence rules** (deterministic, versioned): e.g. `resting_hr ↑ moderate` + `respiratory_rate ↑` + `skin_temp ↑` within a window → candidate `physiological_stress/possible_illness` event (advisory only, never a diagnosis).
- **Persistence:** transient single-reading spikes are suppressed; require persistence over N readings/period (configurable) before raising an event, except for configured acute red-flags.
- **Severity:** derived from the worst contributing deviation, persistence, and confidence.
- Output: `Event` nodes committed to the PSG with `contributing_deviation_ids`. Events are inputs to the LLM (for explanation) and the Policy Engine (for escalation) — they are **not** surfaced to the patient directly.

## 7. Forecast Engine (v1)

- Short-horizon (default 7-day) forecasts on supported metrics' personal baselines.
- v1 methods: per-metric exponential smoothing / Holt-Winters style on the trend+seasonal components already maintained for the baseline; produce point + prediction interval.
- Strictly decision support: forecasts predict *metric trajectories*, never disease. Surfaced through the output contract under `info`, gated by Policy like everything else.
- Behind a `Forecaster` interface so a temporal foundation model can replace it later.

## 8. What this engine must NOT do

- Must not emit patient-facing text (that's the LLM, post-Policy).
- Must not diagnose or name conditions as fact (Event `indicates` edges are advisory candidates).
- Must not learn from sub-threshold readings.
- Must not silently switch from population fallback to personalised without flagging the transition.

## 9. Acceptance (see `11` for targets)

- Deviation precision/recall against labelled events on offline datasets.
- Confidence **calibration** (reliability curve / ECE) — a 0.7 confidence should be right ~70% of the time.
- Population→personalised transition correctness (no premature "personalised" labels).
- Robustness: injected noise/artefacts (failing SQI) must not move the baseline.
