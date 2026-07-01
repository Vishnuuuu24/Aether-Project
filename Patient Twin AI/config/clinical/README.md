# `config/clinical/` — clinician-filled parameter stubs

These files hold the **clinical parameters** the engine must not invent (CLAUDE.md:
*"Never invent clinical content … left as config stubs marked `set with clinical
input`"*). The code implements the *mechanism*; these files supply the *values*.

Every file ships **empty / unset**. Behaviour while a value is unset is fail-safe
and clearly labelled — never a fabricated default.

| File | Purpose | Consumed by | Unset behaviour |
|---|---|---|---|
| `sqi_thresholds.yaml` | Per-metric minimum SQI to enter the baseline (docs/05 §3) | `ai/features` `SqiGate` (T1.2) | Metric never passes the gate → nothing enters its baseline |
| `population_reference_ranges.yaml` | Age/sex cold-start ranges, the *labelled* population fallback (docs/05 §4.1) | `ai/baseline` (T1.3) | No population fallback available → baseline stays unavailable until personalised |

> These are the **population's** numbers, used only as a labelled cold-start fallback.
> They are never presented to a patient as "your normal" (docs/05 §1, §4.1).

## How to fill them
A clinician / signal expert replaces each `# set with clinical input` placeholder
with a concrete value. No code change is needed — the loaders pick up any value
that is present and ignore the ones still unset. Changes to these files are a
human-gated, versioned release (CLAUDE.md: no closed-loop self-modification).
