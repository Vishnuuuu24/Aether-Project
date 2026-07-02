"""Event Engine (docs/05 §6; T2.1).

Combines single-metric deviations into candidate multi-metric **events** using
deterministic, versioned co-occurrence rules with a persistence requirement
(transient spikes are suppressed). Emits `EventCandidate`s — the Patient State
Engine commits them to the PSG. Events are advisory inputs to the LLM and Policy
Engine; they are never surfaced to the patient directly.

Concrete service (no swappable interface — docs/02 §6 lists none for events).
"""

from .engine import EVENT_ENGINE_VERSION, EventEngine
from .rules import (
    CoOccurrenceRule,
    EventRuleSet,
    MetricCondition,
    load_event_rules,
)

__all__ = [
    "EVENT_ENGINE_VERSION",
    "CoOccurrenceRule",
    "EventEngine",
    "EventRuleSet",
    "MetricCondition",
    "load_event_rules",
]
