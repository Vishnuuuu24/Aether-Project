"""Population reference providers (docs/05 §4.1)."""

from __future__ import annotations

from pathlib import Path

from ai.baseline.population import (
    DEFAULT_POPULATION_RANGES_PATH,
    StaticPopulationReferenceProvider,
    YamlPopulationReferenceProvider,
)
from schemas.baseline import PopulationRange


def test_static_provider_returns_range() -> None:
    provider = StaticPopulationReferenceProvider(
        {"heart_rate": PopulationRange(low=60.0, high=100.0, unit="bpm")}
    )
    got = provider.range_for("heart_rate", "resting", age_years=30, sex="female")
    assert got is not None
    assert got.low == 60.0 and got.high == 100.0
    assert provider.range_for("spo2", "resting", age_years=30, sex="female") is None


def test_shipped_yaml_stub_is_unset() -> None:
    # The committed population-ranges stub must not carry fabricated values.
    provider = YamlPopulationReferenceProvider(DEFAULT_POPULATION_RANGES_PATH)
    assert provider.range_for("heart_rate", "resting", age_years=30, sex="female") is None


def test_yaml_provider_matches_filled_entry(tmp_path: Path) -> None:
    cfg = tmp_path / "pop.yaml"
    cfg.write_text(
        "ranges:\n"
        "  heart_rate:\n"
        "    - sex: any\n"
        "      age_min: 18\n"
        "      age_max: 65\n"
        "      context: resting\n"
        "      low: 55\n"
        "      high: 95\n"
        "      unit: bpm\n"
    )
    provider = YamlPopulationReferenceProvider(cfg)
    got = provider.range_for("heart_rate", "resting", age_years=40, sex="male")
    assert got is not None
    assert (got.low, got.high, got.unit) == (55.0, 95.0, "bpm")
    # Out-of-age-range and wrong-context do not match.
    assert provider.range_for("heart_rate", "resting", age_years=70, sex="male") is None
    assert provider.range_for("heart_rate", "active", age_years=40, sex="male") is None
