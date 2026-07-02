"""Co-occurrence rule definitions + loader (docs/05 §6).

The rule MECHANISM (matching, persistence, severity) is code; the rule DEFINITIONS
are CLINICAL config (`config/clinical/event_rules.yaml`), shipped UNSET
(CLAUDE.md: never fabricate clinical content). With no rules configured the engine
raises no events (fail-safe). Tests inject explicit rules to exercise the mechanism.

Rules are versioned and change only via human-gated releases — the engine never
edits them (CLAUDE.md principle 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from schemas.baseline import DeviationMagnitude
from schemas.psg import DeviationDirection

DEFAULT_EVENT_RULES_PATH = Path("config/clinical/event_rules.yaml")


@dataclass(frozen=True)
class MetricCondition:
    """One leg of a co-occurrence rule: a metric deviating a given way, at least a
    given magnitude bucket.
    """

    metric_code: str
    direction: DeviationDirection
    min_magnitude: DeviationMagnitude


@dataclass(frozen=True)
class CoOccurrenceRule:
    id: str
    event_type: str
    window_minutes: int
    persistence_count: int  # min matching deviations per condition (>=1); 1 for red-flags
    conditions: tuple[MetricCondition, ...]
    acute_red_flag: bool = False  # bypass persistence: a single reading may raise it


@dataclass(frozen=True)
class EventRuleSet:
    rules: tuple[CoOccurrenceRule, ...]
    version: str = "unset"


def load_event_rules(path: Path | None = None) -> EventRuleSet:
    """Load co-occurrence rules. Missing file or all-unset stub => empty ruleset."""
    target = path or DEFAULT_EVENT_RULES_PATH
    if not target.exists():
        return EventRuleSet(rules=(), version="unset")
    raw: Any = yaml.safe_load(target.read_text()) or {}
    if not isinstance(raw, dict):
        return EventRuleSet(rules=(), version="unset")
    version = str(raw.get("version") or "unset")
    entries = raw.get("rules")
    if not isinstance(entries, list):
        return EventRuleSet(rules=(), version=version)
    rules: list[CoOccurrenceRule] = []
    for entry in entries:
        rule = _parse_rule(entry)
        if rule is not None:
            rules.append(rule)
    return EventRuleSet(rules=tuple(rules), version=version)


def _parse_rule(entry: Any) -> CoOccurrenceRule | None:
    if not isinstance(entry, dict):
        return None
    required = ("id", "event_type", "window_minutes", "persistence_count", "conditions")
    if any(entry.get(k) is None for k in required):
        return None  # partially-unset stub row
    conditions = _parse_conditions(entry.get("conditions"))
    if not conditions:
        return None
    return CoOccurrenceRule(
        id=str(entry["id"]),
        event_type=str(entry["event_type"]),
        window_minutes=int(entry["window_minutes"]),
        persistence_count=int(entry["persistence_count"]),
        conditions=conditions,
        acute_red_flag=bool(entry.get("acute_red_flag", False)),
    )


def _parse_conditions(raw: Any) -> tuple[MetricCondition, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[MetricCondition] = []
    for cond in raw:
        if not isinstance(cond, dict):
            continue
        metric, direction, magnitude = (
            cond.get("metric_code"),
            cond.get("direction"),
            cond.get("min_magnitude"),
        )
        if metric is None or direction is None or magnitude is None:
            continue
        out.append(
            MetricCondition(
                metric_code=str(metric),
                direction=DeviationDirection(str(direction)),
                min_magnitude=DeviationMagnitude(str(magnitude)),
            )
        )
    return tuple(out)
