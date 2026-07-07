"""Promotion recommendation — advisory only, non-mutating (docs/16 Sprint 10; CLAUDE.md §5)."""

from __future__ import annotations

import json
from pathlib import Path

from ai.training.promotion import (
    Bar,
    evaluate_promotion,
    write_promotion_recommendation,
)


def test_bar_direction() -> None:
    assert Bar("mae", 6.5, 11.0, higher_is_better=False).passed  # lower is better
    assert not Bar("mae", 12.0, 11.0, higher_is_better=False).passed
    assert Bar("acc", 0.83, 0.64, higher_is_better=True).passed
    assert not Bar("acc", 0.60, 0.64, higher_is_better=True).passed


def test_recommends_only_when_all_bars_pass() -> None:
    rec = evaluate_promotion("m@abc", [
        Bar("vs_dsp", 6.5, 11.0, higher_is_better=False),
        Bar("vs_linear", 6.5, 16.9, higher_is_better=False),
    ])
    assert rec.recommended
    assert "beats all 2" in rec.rationale


def test_not_recommended_when_a_bar_fails() -> None:
    rec = evaluate_promotion("m@abc", [
        Bar("vs_dsp", 6.5, 11.0, higher_is_better=False),
        Bar("vs_linear", 20.0, 16.9, higher_is_better=False),
    ])
    assert not rec.recommended
    assert "does not beat" in rec.rationale


def test_empty_bars_is_not_recommended() -> None:
    assert not evaluate_promotion("m@abc", []).recommended


def test_write_recommendation_is_advisory_file_only(tmp_path: Path) -> None:
    rec = evaluate_promotion("m@abc", [Bar("acc", 0.83, 0.64, higher_is_better=True)])
    path = write_promotion_recommendation(rec, tmp_path)
    assert path == tmp_path / "promotion.json"
    data = json.loads(path.read_text())
    assert data["recommended"] is True
    assert "human gate required" in data["decided_by"]
    assert data["bars"][0]["passed"] is True
