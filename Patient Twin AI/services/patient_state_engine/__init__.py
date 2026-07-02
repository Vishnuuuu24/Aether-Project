"""Patient State Engine (docs/02 §4, docs/04 §3, §5; T1.4).

Owns the Patient State Graph (PSG) — the single source of truth. It validates and
commits engine outputs (baselines, deviations) as **versioned, append-only** nodes,
audits every mutation, and builds the **consent-scoped projection** (the only thing
the LLM may ever see — no raw signals).

This is a concrete service, not a swappable interface: unlike BaselineEngine /
FeatureExtractor there is no deferred DL variant of the state engine (docs/02 §6).
"""
