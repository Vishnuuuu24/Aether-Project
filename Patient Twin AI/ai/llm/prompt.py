"""Prompt assembly for the gateway (docs/06 §1-2, docs/04 §5-6).

Turns the consent-scoped `PSGProjection` + retrieved evidence + the user's query into
a (system, user) message pair. Design rules baked in here:

  - The model is told, explicitly and repeatedly, that it PROPOSES and never decides;
    a deterministic Policy Engine is the real gate (CLAUDE.md principle 1).
  - It may only use the enumerated facts/passages, each tagged with a stable ref, and
    must cite the ref for every claim. Refs come from `ai.grounding` so the Policy
    grounding check validates against the exact same set.
  - It must abstain when evidence is insufficient (abstention is a correct outcome),
    and must never diagnose, prescribe, or change medications.

The user message contains only projected, consent-scoped data — no raw signals.
"""

from __future__ import annotations

from ai.grounding import kb_ref, psg_facts
from schemas.psg import PSGProjection
from schemas.retrieval import EvidenceChunk

SYSTEM_PROMPT = """\
You are the explanation layer of a Patient Copilot. You do NOT diagnose, prescribe, \
dose, or recommend medication changes, and you never contradict a recorded clinician \
instruction. A separate deterministic Policy Engine reviews everything you output and \
will suppress it if you break these rules — so proposing anything unsafe only wastes \
the turn.

You may use ONLY the facts and passages provided below, each labelled with a [ref]. \
Every claim in your message MUST cite at least one [ref] via the evidence list, using \
the ref string exactly as given. Do not invent refs, facts, numbers, or citations. \
If the provided evidence is insufficient to answer safely, abstain: set the message \
to a brief explanation that you cannot answer this from the available data, choose \
recommended_action accordingly, and keep confidence low.

Respond with a single JSON object matching the required schema. Do not include any \
text outside the JSON."""


def _evidence_block(projection: PSGProjection, evidence: list[EvidenceChunk]) -> str:
    lines: list[str] = ["PATIENT STATE FACTS (from the Patient State Graph):"]
    facts = psg_facts(projection)
    if facts:
        for ref, text in facts.items():
            lines.append(f"  [{ref}] {text}")
    else:
        lines.append("  (none in scope)")
    lines.append("")
    lines.append("KNOWLEDGE-BASE PASSAGES (retrieved):")
    if evidence:
        for chunk in evidence:
            section = f" ({chunk.section})" if chunk.section else ""
            lines.append(f"  [{kb_ref(chunk)}]{section} {chunk.text}")
    else:
        lines.append("  (no supporting evidence retrieved)")
    return "\n".join(lines)


def build_user_message(
    *, query: str, projection: PSGProjection, evidence: list[EvidenceChunk], locale: str
) -> str:
    demo = (
        f"Patient: age {projection.patient_age_years}, "
        f"sex-at-birth {projection.patient_sex_at_birth}."
    )
    return (
        f"{demo}\n"
        f"Locale: {locale}\n\n"
        f"{_evidence_block(projection, evidence)}\n\n"
        f"USER QUESTION:\n{query}\n\n"
        "Answer using only the [ref]-tagged facts and passages above. Cite every ref "
        "you rely on in the evidence list. Abstain if they are insufficient."
    )
