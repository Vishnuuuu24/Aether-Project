"""Evidence-ref derivation shared by the LLM prompt and the Policy grounding gate.

Grounding is mechanical, not model-judged (docs/06 §2.2): every factual claim the
model makes must cite a ref, and every cited ref must be one we actually handed it.
For that check to be deterministic, the *set of allowed refs* must be derived the
same way in two places — when building the prompt (ai.llm.prompt) and when verifying
the proposal (services.policy_engine). This module is that single source of truth, so
the two never drift.

Refs are opaque, stable strings:
  - KB passages:  ``kb:<chunk_id>``
  - PSG facts:    ``psg:<kind>:<key>``  (baseline/deviation/event/condition/…)

Nothing here reads raw signals — it only sees the already-projected `PSGProjection`
and retrieved `EvidenceChunk`s.
"""

from __future__ import annotations

from schemas.psg import PSGProjection
from schemas.retrieval import EvidenceChunk


def kb_ref(chunk: EvidenceChunk) -> str:
    return f"kb:{chunk.chunk_id}"


def psg_facts(projection: PSGProjection) -> dict[str, str]:
    """Map each citable PSG fact to a stable ref -> human-readable fact string.

    The string is what gets shown to the model next to the ref and what the grounding
    check treats as the fact's content. Order is deterministic.
    """
    facts: dict[str, str] = {}
    for b in projection.baselines:
        kind = "population baseline" if b.is_population_fallback else "personal baseline"
        facts[f"psg:baseline:{b.metric_code.value}:{b.context.value}"] = (
            f"{kind} for {b.metric_code.value} ({b.context.value}): center {b.center}, "
            f"dispersion {b.dispersion}, confidence {b.confidence}"
        )
    for i, d in enumerate(projection.recent_deviations):
        facts[f"psg:deviation:{i}"] = (
            f"{d.metric_code.value} deviated {d.direction.value} (z_robust {d.z_robust}, "
            f"magnitude {d.magnitude}) at {d.ts.isoformat()}"
        )
    for i, e in enumerate(projection.active_events):
        facts[f"psg:event:{i}"] = (
            f"active event '{e.type}' severity {e.severity.value} since {e.onset_ts.isoformat()}"
        )
    for c in projection.conditions:
        facts[f"psg:condition:{c.snomed_code}"] = f"condition {c.display} ({c.status})"
    for m in projection.medications:
        facts[f"psg:medication:{m.rxnorm_code}"] = f"medication {m.display} ({m.status})"
    for a in projection.allergies:
        facts[f"psg:allergy:{a.substance}"] = f"allergy to {a.substance} (reaction {a.reaction})"
    for o in projection.recent_observations:
        facts[f"psg:observation:{o.loinc_code}"] = (
            f"observation {o.display}: {o.value} {o.unit} at {o.ts.isoformat()}"
        )
    for f in projection.latest_forecasts:
        facts[f"psg:forecast:{f.metric_code.value}"] = (
            f"forecast for {f.metric_code.value} over {f.horizon_days}d: points {f.points}"
        )
    return facts


def allowed_refs(projection: PSGProjection, evidence: list[EvidenceChunk]) -> set[str]:
    """The complete set of refs the model is permitted to cite for this query."""
    return {kb_ref(c) for c in evidence} | set(psg_facts(projection))
