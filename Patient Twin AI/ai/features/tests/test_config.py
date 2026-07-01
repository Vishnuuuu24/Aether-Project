"""load_sqi_thresholds: shipped stub is empty; explicit values load; missing file => {}."""

from __future__ import annotations

from pathlib import Path

from ai.features.config import DEFAULT_SQI_THRESHOLDS_PATH, load_sqi_thresholds


def test_shipped_stub_is_all_unset() -> None:
    # The committed clinical stub must not carry fabricated thresholds.
    thresholds = load_sqi_thresholds(DEFAULT_SQI_THRESHOLDS_PATH)
    assert thresholds == {}


def test_missing_file_returns_empty() -> None:
    assert load_sqi_thresholds(Path("config/clinical/does_not_exist.yaml")) == {}


def test_loads_only_set_values(tmp_path: Path) -> None:
    cfg = tmp_path / "sqi.yaml"
    cfg.write_text(
        "thresholds:\n  heart_rate: 0.8\n  steps: 0.5\n  spo2:\n",  # spo2 unset
    )
    thresholds = load_sqi_thresholds(cfg)
    assert thresholds == {"heart_rate": 0.8, "steps": 0.5}
