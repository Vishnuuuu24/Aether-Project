"""Strict validators for each clinical config (T8.3).

Each validator takes the raw, YAML-parsed structure and returns the number of
entries that are actually SET, raising `ClinicalConfigError` on anything PRESENT
but malformed. The contract everywhere:

  * absent / null  -> UNSET, counted as 0, never an error (fail-safe by design);
  * present + valid -> counted;
  * present + wrong -> `ClinicalConfigError` (wrong type, out-of-range, bad enum,
    or a partially-filled row a clinician half-completed).

These validate STRUCTURE and value RANGES only — never clinical correctness. They
do not invent content (CLAUDE.md); an all-unset stub validates cleanly at 0 set.
"""

from __future__ import annotations

from typing import Any

from schemas.baseline import DeviationMagnitude
from schemas.output_contract import OutputType, RecommendedAction
from schemas.psg import DeviationDirection, EventSeverity
from schemas.reading import MeasurementContext, MetricCode

from .errors import ClinicalConfigError

_KNOWN_METRICS = {m.value for m in MetricCode}
_CODING_ENTITY_TYPES = frozenset({"condition", "medication", "observation", "allergy"})
_SEXES = frozenset({"male", "female", "any"})
_CONTEXTS = {c.value for c in MeasurementContext} | {"any"}
# A red flag must escalate — only these actions are meaningful for one (docs/06).
_ESCALATING_ACTIONS = frozenset({RecommendedAction.SEEK_CARE, RecommendedAction.SEEK_URGENT_CARE})


def _number_in_range(section: str, key: str, value: Any, lo: float, hi: float) -> float:
    # bool is an int subclass but is never a valid numeric clinical value.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ClinicalConfigError(
            section, f"expected a number, got {type(value).__name__}", key=key
        )
    v = float(value)
    if not lo <= v <= hi:
        raise ClinicalConfigError(section, f"value {v} out of range [{lo}, {hi}]", key=key)
    return v


def _enum(section: str, key: str, value: Any, allowed: frozenset[str] | set[str]) -> str:
    s = str(value)
    if s not in allowed:
        raise ClinicalConfigError(
            section, f"invalid value {value!r}; allowed: {sorted(allowed)}", key=key
        )
    return s


def _section_dict(raw: Any, key: str) -> dict[str, Any]:
    """Return `raw[key]` when it's a mapping; {} when unset; error when wrong type."""
    if not isinstance(raw, dict):
        return {}
    value = raw.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ClinicalConfigError(key, "expected a mapping section")
    return value


def validate_sqi_thresholds(raw: Any) -> int:
    section = _section_dict(raw, "thresholds")
    n_set = 0
    for metric, value in section.items():
        if metric not in _KNOWN_METRICS:
            raise ClinicalConfigError(
                "sqi_thresholds", f"unknown metric {metric!r}", key=str(metric)
            )
        if value is None:
            continue
        _number_in_range("sqi_thresholds", str(metric), value, 0.0, 1.0)
        n_set += 1
    return n_set


def validate_coding_thresholds(raw: Any) -> int:
    section = _section_dict(raw, "thresholds")
    n_set = 0
    for entity, value in section.items():
        if entity not in _CODING_ENTITY_TYPES:
            raise ClinicalConfigError(
                "coding_thresholds", f"unknown entity type {entity!r}", key=str(entity)
            )
        if value is None:
            continue
        _number_in_range("coding_thresholds", str(entity), value, 0.0, 1.0)
        n_set += 1
    return n_set


def validate_population_ranges(raw: Any) -> int:
    section = _section_dict(raw, "ranges")
    n_set = 0
    for metric, entries in section.items():
        if metric not in _KNOWN_METRICS:
            raise ClinicalConfigError(
                "population_ranges", f"unknown metric {metric!r}", key=str(metric)
            )
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ClinicalConfigError(
                "population_ranges", "expected a list of ranges", key=str(metric)
            )
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ClinicalConfigError(
                    "population_ranges", "range entry must be a mapping", key=f"{metric}[{i}]"
                )
            n_set += _validate_range_entry(str(metric), i, entry)
    return n_set


def _validate_range_entry(metric: str, i: int, entry: dict[str, Any]) -> int:
    key = f"{metric}[{i}]"
    low, high, unit = entry.get("low"), entry.get("high"), entry.get("unit")
    provided = [x for x in (low, high, unit) if x is not None]
    if not provided:
        return 0  # a fully-unset stub row is inert, not an error
    if low is None or high is None or unit is None:
        raise ClinicalConfigError(
            "population_ranges", "low, high and unit must all be set together", key=key
        )
    lo = _number_in_range("population_ranges", key, low, -1e9, 1e9)
    hi = _number_in_range("population_ranges", key, high, -1e9, 1e9)
    if hi < lo:
        raise ClinicalConfigError("population_ranges", f"high {hi} < low {lo}", key=key)
    if not str(unit).strip():
        raise ClinicalConfigError("population_ranges", "unit must not be blank", key=key)
    if entry.get("sex") is not None:
        _enum("population_ranges", key, entry["sex"], _SEXES)
    if entry.get("context") is not None:
        _enum("population_ranges", key, entry["context"], _CONTEXTS)
    a_min, a_max = entry.get("age_min"), entry.get("age_max")
    if a_min is not None and a_max is not None and float(a_max) < float(a_min):
        raise ClinicalConfigError("population_ranges", "age_max < age_min", key=key)
    return 1


def validate_event_rules(raw: Any) -> int:
    if raw is None:
        return 0
    if not isinstance(raw, dict):
        raise ClinicalConfigError("event_rules", "expected a mapping document")
    entries = raw.get("rules")
    if entries is None:
        return 0
    if not isinstance(entries, list):
        raise ClinicalConfigError("event_rules", "'rules' must be a list")
    n_set = 0
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ClinicalConfigError("event_rules", "rule must be a mapping", key=f"[{i}]")
        n_set += _validate_event_rule(i, entry)
    return n_set


def _validate_event_rule(i: int, entry: dict[str, Any]) -> int:
    key = f"rules[{i}]"
    required = ("id", "event_type", "window_minutes", "persistence_count", "conditions")
    present = [k for k in required if entry.get(k) is not None]
    if not present:
        return 0
    missing = [k for k in required if entry.get(k) is None]
    if missing:
        raise ClinicalConfigError("event_rules", f"missing required fields {missing}", key=key)
    if int(entry["window_minutes"]) <= 0:
        raise ClinicalConfigError("event_rules", "window_minutes must be > 0", key=key)
    if int(entry["persistence_count"]) < 1:
        raise ClinicalConfigError("event_rules", "persistence_count must be >= 1", key=key)
    conditions = entry["conditions"]
    if not isinstance(conditions, list) or not conditions:
        raise ClinicalConfigError("event_rules", "conditions must be a non-empty list", key=key)
    for j, cond in enumerate(conditions):
        ckey = f"{key}.conditions[{j}]"
        if not isinstance(cond, dict):
            raise ClinicalConfigError("event_rules", "condition must be a mapping", key=ckey)
        _enum("event_rules", ckey, cond.get("metric_code"), _KNOWN_METRICS)
        _enum("event_rules", ckey, cond.get("direction"), {d.value for d in DeviationDirection})
        _enum("event_rules", ckey, cond.get("min_magnitude"), {m.value for m in DeviationMagnitude})
    return 1


def validate_policy_rules(raw: Any) -> int:
    if raw is None:
        return 0
    if not isinstance(raw, dict):
        raise ClinicalConfigError("policy_rules", "expected a mapping document")
    n_set = 0
    n_set += _validate_red_flags(raw.get("red_flags"))
    n_set += _validate_confidence_thresholds(raw.get("confidence_thresholds"))
    n_set += _validate_prohibited_terms(raw.get("prohibited_terms"))
    return n_set


def _validate_red_flags(raw: Any) -> int:
    if raw is None:
        return 0
    if not isinstance(raw, list):
        raise ClinicalConfigError("policy_rules", "'red_flags' must be a list")
    n_set = 0
    for i, entry in enumerate(raw):
        key = f"red_flags[{i}]"
        if not isinstance(entry, dict):
            raise ClinicalConfigError("policy_rules", "red flag must be a mapping", key=key)
        if entry.get("id") is None and entry.get("action") is None:
            continue  # fully-unset stub row
        if entry.get("id") is None or entry.get("action") is None:
            raise ClinicalConfigError("policy_rules", "red flag needs both id and action", key=key)
        action = _enum("policy_rules", key, entry["action"], {a.value for a in RecommendedAction})
        if RecommendedAction(action) not in _ESCALATING_ACTIONS:
            raise ClinicalConfigError(
                "policy_rules", "red flag action must be seek_care or seek_urgent_care", key=key
            )
        if entry.get("min_event_severity") is not None:
            _enum(
                "policy_rules", key, entry["min_event_severity"], {s.value for s in EventSeverity}
            )
        types = entry.get("any_active_event_type")
        if types is not None and not isinstance(types, list):
            raise ClinicalConfigError(
                "policy_rules", "any_active_event_type must be a list", key=key
            )
        n_set += 1
    return n_set


def _validate_confidence_thresholds(raw: Any) -> int:
    if raw is None:
        return 0
    if not isinstance(raw, dict):
        raise ClinicalConfigError("policy_rules", "'confidence_thresholds' must be a mapping")
    n_set = 0
    for otype, value in raw.items():
        _enum(
            "policy_rules.confidence_thresholds", str(otype), otype, {o.value for o in OutputType}
        )
        if value is None:
            continue
        _number_in_range("policy_rules.confidence_thresholds", str(otype), value, 0.0, 1.0)
        n_set += 1
    return n_set


def _validate_prohibited_terms(raw: Any) -> int:
    if raw is None:
        return 0
    if not isinstance(raw, list):
        raise ClinicalConfigError("policy_rules", "'prohibited_terms' must be a list")
    n_set = 0
    for i, term in enumerate(raw):
        if term is None:
            continue
        if not isinstance(term, str) or not term.strip():
            raise ClinicalConfigError(
                "policy_rules",
                "prohibited term must be a non-empty string",
                key=f"prohibited_terms[{i}]",
            )
        n_set += 1
    return n_set


def validate_kb_content(raw: Any) -> int:
    """Validate a KB-content manifest (passages). UNSET => inert empty KB.

    Structure only: each passage that is present must carry a non-blank id, source
    and text. The passage TEXT itself is curated clinical content and stays a
    clinician/curation deliverable — never fabricated here (CLAUDE.md).
    """
    if raw is None:
        return 0
    if not isinstance(raw, dict):
        raise ClinicalConfigError("kb_content", "expected a mapping document")
    passages = raw.get("passages")
    if passages is None:
        return 0
    if not isinstance(passages, list):
        raise ClinicalConfigError("kb_content", "'passages' must be a list")
    n_set = 0
    for i, entry in enumerate(passages):
        key = f"passages[{i}]"
        if not isinstance(entry, dict):
            raise ClinicalConfigError("kb_content", "passage must be a mapping", key=key)
        if all(entry.get(f) is None for f in ("id", "source", "text")):
            continue  # fully-unset stub row
        for field in ("id", "source", "text"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ClinicalConfigError(
                    "kb_content", f"passage {field} must be a non-blank string", key=key
                )
        codes = entry.get("codes")
        if codes is not None and not isinstance(codes, list):
            raise ClinicalConfigError("kb_content", "codes must be a list", key=key)
        n_set += 1
    return n_set
