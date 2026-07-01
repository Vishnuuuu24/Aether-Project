"""Normalisation: raw adapter rows → validated `Reading` (docs/02 §2, docs/04 §2).

The single point where any adapter's output becomes a canonical Reading, so
validation and field-level rejection live in ONE place (docs/07 §3). Rules:

- Missing/malformed metadata → rejected with field-level errors (never silently
  dropped or coerced).
- SQI is NOT computed here (that is the SQI service, T1.2). A reading with no sqi
  is marked "unknown" (0.0 — below any threshold) until the SQI stage scores it.
- `included_in_baseline` is forced False: it is set by the SQI gate, never by the
  sender (docs/04 §2).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from schemas.reading import IngestRejection, Reading

# Placeholder SQI for "not yet scored"; the SQI service overwrites it (T1.2).
UNKNOWN_SQI = 0.0


class ReadingRejected(Exception):
    """A single raw reading failed validation. Carries field-level errors."""

    def __init__(self, errors: list[IngestRejection]) -> None:
        super().__init__("reading rejected")
        self.errors = errors


@dataclass
class NormalisationResult:
    # (source_index, reading) so downstream consent rejections keep the original index.
    accepted: list[tuple[int, Reading]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)  # {"index", "errors"}


def _prepare(raw: Mapping[str, Any], *, default_adapter: str) -> dict[str, Any]:
    data = dict(raw)
    # The sender may not set baseline membership — the SQI gate owns it.
    data.pop("included_in_baseline", None)
    data["included_in_baseline"] = False
    data.setdefault("ingest_adapter", default_adapter)
    if data.get("sqi") in (None, ""):
        data["sqi"] = UNKNOWN_SQI
    return data


def _errors_from(exc: ValidationError) -> list[IngestRejection]:
    errors: list[IngestRejection] = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        errors.append(IngestRejection(field=loc, issue=err["msg"]))
    return errors


def normalise_one(raw: Mapping[str, Any], *, default_adapter: str) -> Reading:
    """Validate one raw reading into a Reading, or raise ReadingRejected."""
    data = _prepare(raw, default_adapter=default_adapter)
    try:
        return Reading.model_validate(data)
    except ValidationError as exc:
        raise ReadingRejected(_errors_from(exc)) from exc


def normalise_batch(
    items: Iterable[Mapping[str, Any]], *, default_adapter: str
) -> NormalisationResult:
    """Validate a batch, collecting accepted readings and per-item field errors."""
    result = NormalisationResult()
    for index, raw in enumerate(items):
        try:
            result.accepted.append((index, normalise_one(raw, default_adapter=default_adapter)))
        except ReadingRejected as rejected:
            result.rejections.append(
                {"index": index, "errors": [e.model_dump() for e in rejected.errors]}
            )
    return result
