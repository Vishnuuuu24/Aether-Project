"""Loader for per-entity-type coding confidence thresholds
(`config/clinical/coding_thresholds.yaml`; docs/04 §4).

A coded entity is `committed` only when its confidence >= the threshold for its
entity type; otherwise it stays `proposed` (awaits human confirmation). Thresholds
are CLINICAL config, shipped UNSET (CLAUDE.md) — so the gate is fail-safe: with no
threshold set, everything stays `proposed` and nothing is auto-committed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CODING_THRESHOLDS_PATH = Path("config/clinical/coding_thresholds.yaml")


def load_coding_thresholds(path: Path | None = None) -> dict[str, float]:
    """Load per-entity-type confidence thresholds. Missing file / all-unset => {}."""
    target = path or DEFAULT_CODING_THRESHOLDS_PATH
    if not target.exists():
        return {}
    raw: Any = yaml.safe_load(target.read_text()) or {}
    section: Any = raw.get("thresholds") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        return {}
    resolved: dict[str, float] = {}
    for entity_type, value in section.items():
        if value is None:
            continue
        resolved[str(entity_type)] = float(value)
    return resolved
