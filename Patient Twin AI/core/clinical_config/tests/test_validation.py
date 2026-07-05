"""T8.3: clinical-config validation — shipped stubs are valid-and-unset; present
values validate; malformed values fail loudly; UNSET stays inert."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.clinical_config import ClinicalConfigError, validate_clinical_configs
from core.clinical_config.validators import (
    validate_coding_thresholds,
    validate_event_rules,
    validate_kb_content,
    validate_policy_rules,
    validate_population_ranges,
    validate_sqi_thresholds,
)
from scripts.validate_clinical_config import _main

_SHIPPED = Path("config/clinical")


def test_shipped_stubs_validate_and_are_all_unset() -> None:
    report = validate_clinical_configs(_SHIPPED)
    assert report.all_unset, "shipped stubs must carry no fabricated clinical values"
    # Every registered section resolves to a report entry.
    names = {s.name for s in report.sections}
    assert names >= {"sqi_thresholds", "policy_rules", "event_rules", "kb_content"}


def test_missing_dir_is_all_unset_not_error(tmp_path: Path) -> None:
    report = validate_clinical_configs(tmp_path)  # empty dir → every file missing
    assert report.all_unset
    assert all(not s.present for s in report.sections)


# -- SQI thresholds ----------------------------------------------------------


def test_sqi_valid_value_counts() -> None:
    assert validate_sqi_thresholds({"thresholds": {"heart_rate": 0.8, "spo2": None}}) == 1


def test_sqi_out_of_range_raises() -> None:
    with pytest.raises(ClinicalConfigError, match="out of range"):
        validate_sqi_thresholds({"thresholds": {"heart_rate": 1.5}})


def test_sqi_unknown_metric_raises() -> None:
    with pytest.raises(ClinicalConfigError, match="unknown metric"):
        validate_sqi_thresholds({"thresholds": {"blood_pressure": 0.5}})


def test_sqi_non_numeric_raises() -> None:
    with pytest.raises(ClinicalConfigError, match="expected a number"):
        validate_sqi_thresholds({"thresholds": {"heart_rate": "high"}})


# -- coding thresholds -------------------------------------------------------


def test_coding_unknown_entity_raises() -> None:
    with pytest.raises(ClinicalConfigError, match="unknown entity type"):
        validate_coding_thresholds({"thresholds": {"procedure": 0.5}})


# -- population ranges -------------------------------------------------------


def test_population_partial_row_raises() -> None:
    raw = {"ranges": {"heart_rate": [{"low": 60, "unit": "bpm"}]}}  # missing high
    with pytest.raises(ClinicalConfigError, match="must all be set together"):
        validate_population_ranges(raw)


def test_population_high_below_low_raises() -> None:
    raw = {"ranges": {"heart_rate": [{"low": 100, "high": 60, "unit": "bpm"}]}}
    with pytest.raises(ClinicalConfigError, match="high 60.0 < low 100.0"):
        validate_population_ranges(raw)


def test_population_valid_entry_counts() -> None:
    raw = {"ranges": {"heart_rate": [{"sex": "any", "low": 50, "high": 90, "unit": "bpm"}]}}
    assert validate_population_ranges(raw) == 1


def test_population_fully_unset_row_is_inert() -> None:
    assert validate_population_ranges({"ranges": {"heart_rate": None}}) == 0


# -- event rules -------------------------------------------------------------


def test_event_rule_missing_field_raises() -> None:
    raw = {"rules": [{"id": "r1", "event_type": "x"}]}  # missing window/persistence/conditions
    with pytest.raises(ClinicalConfigError, match="missing required fields"):
        validate_event_rules(raw)


def test_event_rule_bad_direction_raises() -> None:
    raw = {
        "rules": [
            {
                "id": "r1",
                "event_type": "x",
                "window_minutes": 10,
                "persistence_count": 2,
                "conditions": [
                    {"metric_code": "heart_rate", "direction": "sideways", "min_magnitude": "mild"}
                ],
            }
        ]
    }
    with pytest.raises(ClinicalConfigError, match="invalid value"):
        validate_event_rules(raw)


def test_event_rules_empty_is_inert() -> None:
    assert validate_event_rules({"version": "unset", "rules": []}) == 0


# -- policy rules ------------------------------------------------------------


def test_policy_red_flag_non_escalating_action_raises() -> None:
    raw = {"red_flags": [{"id": "r1", "action": "monitor"}]}
    with pytest.raises(ClinicalConfigError, match="seek_care or seek_urgent_care"):
        validate_policy_rules(raw)


def test_policy_confidence_threshold_out_of_range_raises() -> None:
    with pytest.raises(ClinicalConfigError, match="out of range"):
        validate_policy_rules({"confidence_thresholds": {"info": 2.0}})


def test_policy_unknown_output_type_raises() -> None:
    with pytest.raises(ClinicalConfigError, match="invalid value"):
        validate_policy_rules({"confidence_thresholds": {"summary": 0.5}})


def test_policy_blank_prohibited_term_raises() -> None:
    with pytest.raises(ClinicalConfigError, match="non-empty string"):
        validate_policy_rules({"prohibited_terms": ["diagnose", "  "]})


def test_policy_valid_full_ruleset_counts() -> None:
    raw = {
        "red_flags": [{"id": "r1", "action": "seek_urgent_care", "min_event_severity": "high"}],
        "confidence_thresholds": {"info": 0.6, "guidance": 0.8},
        "prohibited_terms": ["you have", "prescribe"],
    }
    assert validate_policy_rules(raw) == 5  # 1 red flag + 2 thresholds + 2 terms


# -- KB content --------------------------------------------------------------


def test_kb_blank_text_raises() -> None:
    raw = {"passages": [{"id": "p1", "source": "guidelines", "text": "   "}]}
    with pytest.raises(ClinicalConfigError, match="non-blank string"):
        validate_kb_content(raw)


def test_kb_valid_passage_counts() -> None:
    raw = {"passages": [{"id": "p1", "source": "guidelines", "text": "hydration matters"}]}
    assert validate_kb_content(raw) == 1


def test_kb_empty_is_inert() -> None:
    assert validate_kb_content({"passages": []}) == 0


# -- CLI ---------------------------------------------------------------------


def test_cli_ok_on_shipped_stubs() -> None:
    assert _main([]) == 0
    assert _main(["--json"]) == 0


def test_cli_fails_on_malformed(tmp_path: Path) -> None:
    (tmp_path / "sqi_thresholds.yaml").write_text("thresholds:\n  heart_rate: 9.9\n")
    assert _main(["--dir", str(tmp_path)]) == 2
