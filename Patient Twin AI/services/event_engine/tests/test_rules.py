"""load_event_rules: shipped stub is empty; explicit rules parse; missing file => empty."""

from __future__ import annotations

from pathlib import Path

from schemas.baseline import DeviationMagnitude
from schemas.psg import DeviationDirection
from services.event_engine.rules import DEFAULT_EVENT_RULES_PATH, load_event_rules


def test_shipped_stub_has_no_rules() -> None:
    # The committed clinical stub must not carry fabricated co-occurrence rules.
    ruleset = load_event_rules(DEFAULT_EVENT_RULES_PATH)
    assert ruleset.rules == ()


def test_missing_file_returns_empty() -> None:
    assert load_event_rules(Path("config/clinical/does_not_exist.yaml")).rules == ()


def test_parses_filled_rule(tmp_path: Path) -> None:
    cfg = tmp_path / "event_rules.yaml"
    cfg.write_text(
        "version: r1\n"
        "rules:\n"
        "  - id: stress-1\n"
        "    event_type: physiological_stress\n"
        "    window_minutes: 120\n"
        "    persistence_count: 2\n"
        "    conditions:\n"
        "      - metric_code: heart_rate\n"
        "        direction: up\n"
        "        min_magnitude: moderate\n"
        "      - metric_code: respiratory_rate\n"
        "        direction: up\n"
        "        min_magnitude: mild\n"
    )
    ruleset = load_event_rules(cfg)
    assert ruleset.version == "r1"
    assert len(ruleset.rules) == 1
    rule = ruleset.rules[0]
    assert rule.id == "stress-1"
    assert rule.window_minutes == 120
    assert rule.persistence_count == 2
    assert len(rule.conditions) == 2
    assert rule.conditions[0].direction is DeviationDirection.UP
    assert rule.conditions[0].min_magnitude is DeviationMagnitude.MODERATE


def test_partially_unset_rule_skipped(tmp_path: Path) -> None:
    cfg = tmp_path / "event_rules.yaml"
    cfg.write_text(
        "rules:\n"
        "  - id: incomplete\n"
        "    event_type: x\n"
        "    window_minutes:\n"  # unset -> whole rule skipped
        "    persistence_count: 2\n"
        "    conditions:\n"
        "      - metric_code: heart_rate\n"
        "        direction: up\n"
        "        min_magnitude: moderate\n"
    )
    assert load_event_rules(cfg).rules == ()
