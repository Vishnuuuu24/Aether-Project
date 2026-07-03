"""Policy ruleset loader: the shipped clinical stub is UNSET => inert config, and the
structural safety rules still run on top of it (docs/06; CLAUDE.md)."""

from __future__ import annotations

from pathlib import Path

from schemas.output_contract import OutputType, RecommendedAction
from schemas.psg import EventSeverity
from services.policy_engine.rules import load_policy_rules


def test_shipped_stub_loads_as_inert() -> None:
    rules = load_policy_rules(Path("config/clinical/policy_rules.yaml"))
    assert rules.version == "unset"
    assert rules.red_flags == ()
    assert rules.confidence_thresholds == {}
    assert rules.prohibited_terms == ()


def test_missing_file_is_empty_ruleset() -> None:
    rules = load_policy_rules(Path("config/clinical/does_not_exist.yaml"))
    assert rules.red_flags == ()
    assert rules.prohibited_terms == ()


def test_parses_a_populated_ruleset(tmp_path: Path) -> None:
    yaml_text = """
version: policy-v1
red_flags:
  - id: rf_spo2
    action: seek_urgent_care
    any_active_event_type: [hypoxemia]
    min_event_severity: high
  - id: bad_missing_action        # dropped: no action
confidence_thresholds:
  info: 0.6
  guidance: 0.75
prohibited_terms:
  - "you have"
  - "prescribe"
"""
    path = tmp_path / "policy_rules.yaml"
    path.write_text(yaml_text)
    rules = load_policy_rules(path)
    assert rules.version == "policy-v1"
    assert len(rules.red_flags) == 1
    rf = rules.red_flags[0]
    assert rf.id == "rf_spo2"
    assert rf.action == RecommendedAction.SEEK_URGENT_CARE
    assert rf.min_event_severity == EventSeverity.HIGH
    assert rules.confidence_thresholds[OutputType.INFO] == 0.6
    assert "prescribe" in rules.prohibited_terms
