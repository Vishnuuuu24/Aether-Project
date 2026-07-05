"""Aggregate clinical-config validation gate (T8.3).

`validate_clinical_configs()` reads every file under `config/clinical/`, runs its
strict validator, and returns a `ClinicalConfigReport` of what is SET vs UNSET.
It RAISES `ClinicalConfigError` on the first malformed entry — the "clear failure
when malformed" gate — while an all-unset stub validates cleanly (every section at
0 set). Intended to run at service startup / in CI so a config mistake fails loud
instead of silently degrading to fail-safe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ClinicalConfigError
from .validators import (
    validate_coding_thresholds,
    validate_event_rules,
    validate_kb_content,
    validate_policy_rules,
    validate_population_ranges,
    validate_sqi_thresholds,
)

DEFAULT_CLINICAL_CONFIG_DIR = Path("config/clinical")

# (section name, filename, validator). Missing files are reported UNSET, not errors —
# the whole directory is optional and fail-safe when absent.
_REGISTRY: tuple[tuple[str, str, Callable[[Any], int]], ...] = (
    ("sqi_thresholds", "sqi_thresholds.yaml", validate_sqi_thresholds),
    ("coding_thresholds", "coding_thresholds.yaml", validate_coding_thresholds),
    ("population_ranges", "population_reference_ranges.yaml", validate_population_ranges),
    ("event_rules", "event_rules.yaml", validate_event_rules),
    ("policy_rules", "policy_rules.yaml", validate_policy_rules),
    ("kb_content", "kb_content.yaml", validate_kb_content),
)


@dataclass(frozen=True)
class SectionReport:
    name: str
    path: str
    present: bool  # the file exists on disk
    n_set: int  # entries that carry concrete (non-stub) values


@dataclass(frozen=True)
class ClinicalConfigReport:
    sections: tuple[SectionReport, ...]

    @property
    def total_set(self) -> int:
        return sum(s.n_set for s in self.sections)

    @property
    def all_unset(self) -> bool:
        """True when the shipped clinician stubs carry no concrete values yet."""
        return self.total_set == 0


def validate_clinical_configs(
    root: Path | None = None,
) -> ClinicalConfigReport:
    """Validate every clinical config under `root`. Raises `ClinicalConfigError` on the
    first malformed entry; returns a set/unset report when all present files are valid.
    """
    base = root or DEFAULT_CLINICAL_CONFIG_DIR
    sections: list[SectionReport] = []
    for name, filename, validator in _REGISTRY:
        path = base / filename
        if not path.exists():
            sections.append(SectionReport(name=name, path=str(path), present=False, n_set=0))
            continue
        try:
            raw: Any = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:  # malformed YAML is a clear failure
            raise ClinicalConfigError(name, f"invalid YAML: {exc}") from exc
        n_set = validator(raw)
        sections.append(SectionReport(name=name, path=str(path), present=True, n_set=n_set))
    return ClinicalConfigReport(sections=tuple(sections))
