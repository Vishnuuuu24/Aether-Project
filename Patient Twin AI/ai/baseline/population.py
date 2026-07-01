"""Population reference range providers (docs/05 §4.1).

The cold-start fallback: age/sex population ranges, always surfaced with
`is_population_fallback=true`. Values are clinical config in
`config/clinical/population_reference_ranges.yaml`, shipped UNSET — so in v1 the
YAML provider returns None for everything and the baseline stays UNAVAILABLE until
personalised. Tests inject a `StaticPopulationReferenceProvider` to exercise the
fallback path without fabricating clinical numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from schemas.baseline import PopulationRange

DEFAULT_POPULATION_RANGES_PATH = Path("config/clinical/population_reference_ranges.yaml")


@runtime_checkable
class PopulationReferenceProvider(Protocol):
    def range_for(
        self, metric_code: str, context: str, *, age_years: int | None, sex: str | None
    ) -> PopulationRange | None: ...


@dataclass(frozen=True)
class StaticPopulationReferenceProvider:
    """In-memory provider keyed by metric_code. For tests and simple wiring."""

    ranges: dict[str, PopulationRange]

    def range_for(
        self, metric_code: str, context: str, *, age_years: int | None, sex: str | None
    ) -> PopulationRange | None:
        return self.ranges.get(metric_code)


class YamlPopulationReferenceProvider:
    """Loads `config/clinical/population_reference_ranges.yaml` and matches entries
    by sex / age / context. Returns None while the stub is unset — no fabrication.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_POPULATION_RANGES_PATH
        self._ranges = _load_ranges(self._path)

    def range_for(
        self, metric_code: str, context: str, *, age_years: int | None, sex: str | None
    ) -> PopulationRange | None:
        for entry in self._ranges.get(metric_code, []):
            if not _matches(entry, context=context, age_years=age_years, sex=sex):
                continue
            low, high, unit = entry.get("low"), entry.get("high"), entry.get("unit")
            if low is None or high is None or unit is None:
                continue  # partially-unset stub row
            return PopulationRange(low=float(low), high=float(high), unit=str(unit))
        return None


def _load_ranges(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    raw: Any = yaml.safe_load(path.read_text()) or {}
    section: Any = raw.get("ranges") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for metric, entries in section.items():
        if isinstance(entries, list):
            out[str(metric)] = [e for e in entries if isinstance(e, dict)]
    return out


def _matches(
    entry: dict[str, Any], *, context: str, age_years: int | None, sex: str | None
) -> bool:
    entry_sex = entry.get("sex")
    if entry_sex not in (None, "any") and sex is not None and str(entry_sex) != sex:
        return False
    entry_context = entry.get("context")
    if entry_context not in (None, "any") and str(entry_context) != context:
        return False
    if age_years is not None:
        age_min, age_max = entry.get("age_min"), entry.get("age_max")
        if age_min is not None and age_years < float(age_min):
            return False
        if age_max is not None and age_years > float(age_max):
            return False
    return True
