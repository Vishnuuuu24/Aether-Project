"""Policy rule definitions + loader (docs/06 §2).

The rule MECHANISM lives in engine.py; the DEFINITIONS are CLINICAL config
(`config/clinical/policy_rules.yaml`), shipped UNSET (CLAUDE.md). Missing / all-unset
config => an empty ruleset, under which the configurable checks are inert and only the
structural safety rules run. Versioned; changed only via human-gated releases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from schemas.output_contract import OutputType, RecommendedAction
from schemas.psg import EventSeverity

DEFAULT_POLICY_RULES_PATH = Path("config/clinical/policy_rules.yaml")

_SEVERITY_RANK: dict[EventSeverity, int] = {
    EventSeverity.NONE: 0,
    EventSeverity.LOW: 1,
    EventSeverity.MODERATE: 2,
    EventSeverity.HIGH: 3,
}


def severity_rank(severity: EventSeverity) -> int:
    return _SEVERITY_RANK[severity]


@dataclass(frozen=True)
class RedFlagRule:
    """An acute pattern that forces an escalation regardless of LLM output."""

    id: str
    action: RecommendedAction  # seek_care | seek_urgent_care
    any_active_event_type: tuple[str, ...] = ()
    min_event_severity: EventSeverity | None = None


@dataclass(frozen=True)
class PolicyRuleSet:
    version: str = "unset"
    red_flags: tuple[RedFlagRule, ...] = ()
    confidence_thresholds: dict[OutputType, float] = field(default_factory=dict)
    prohibited_terms: tuple[str, ...] = ()


def load_policy_rules(path: Path | None = None) -> PolicyRuleSet:
    target = path or DEFAULT_POLICY_RULES_PATH
    if not target.exists():
        return PolicyRuleSet()
    raw: Any = yaml.safe_load(target.read_text()) or {}
    if not isinstance(raw, dict):
        return PolicyRuleSet()
    return PolicyRuleSet(
        version=str(raw.get("version") or "unset"),
        red_flags=_parse_red_flags(raw.get("red_flags")),
        confidence_thresholds=_parse_thresholds(raw.get("confidence_thresholds")),
        prohibited_terms=_parse_terms(raw.get("prohibited_terms")),
    )


def _parse_red_flags(raw: Any) -> tuple[RedFlagRule, ...]:
    if not isinstance(raw, list):
        return ()
    out: list[RedFlagRule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if entry.get("id") is None or entry.get("action") is None:
            continue  # partially-unset stub row
        try:
            action = RecommendedAction(str(entry["action"]))
        except ValueError:
            continue
        if action not in (RecommendedAction.SEEK_CARE, RecommendedAction.SEEK_URGENT_CARE):
            continue  # a red flag must escalate
        types = entry.get("any_active_event_type") or []
        min_sev = entry.get("min_event_severity")
        out.append(
            RedFlagRule(
                id=str(entry["id"]),
                action=action,
                any_active_event_type=tuple(str(t) for t in types if t is not None),
                min_event_severity=EventSeverity(str(min_sev)) if min_sev is not None else None,
            )
        )
    return tuple(out)


def _parse_thresholds(raw: Any) -> dict[OutputType, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[OutputType, float] = {}
    for key, value in raw.items():
        if value is None:
            continue
        try:
            out[OutputType(str(key))] = float(value)
        except ValueError:
            continue
    return out


def _parse_terms(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(t).lower() for t in raw if t is not None and str(t).strip())
