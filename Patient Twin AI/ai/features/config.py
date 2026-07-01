"""Loader for the clinical SQI-threshold config (`config/clinical/sqi_thresholds.yaml`).

Returns only the metrics that carry a concrete numeric threshold. Unset (null)
entries — the shipped default — are omitted, so `SqiGate` treats them fail-safe
(docs/05 §3). The config file is a clinician-filled stub; this loader never
invents a value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo-relative default. Resolved from the process CWD (the repo root in dev/CI).
DEFAULT_SQI_THRESHOLDS_PATH = Path("config/clinical/sqi_thresholds.yaml")


def load_sqi_thresholds(path: Path | None = None) -> dict[str, float]:
    """Load per-metric SQI thresholds. Missing file or all-unset stub => ``{}``."""
    target = path or DEFAULT_SQI_THRESHOLDS_PATH
    if not target.exists():
        return {}
    raw: Any = yaml.safe_load(target.read_text()) or {}
    section: Any = raw.get("thresholds") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        return {}
    resolved: dict[str, float] = {}
    for metric, value in section.items():
        if value is None:
            continue  # unset stub — awaiting clinical input
        resolved[str(metric)] = float(value)
    return resolved
