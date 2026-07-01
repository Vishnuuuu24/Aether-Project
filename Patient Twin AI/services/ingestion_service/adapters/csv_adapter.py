"""CSV / manual-upload adapter (docs/07 §3).

Header row uses canonical Reading field names; one reading per row, scalar values
only (structured metrics like sleep stages come via the JSON endpoint). Rows are
yielded as canonical dicts and validated by the shared normaliser — a row missing
`unit`/`timestamp`/etc. is rejected there with a field error.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ADAPTER_NAME = "csv"


def _coerce_number(value: str) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _clean(row: dict[str, str | None]) -> dict[str, Any]:
    data: dict[str, Any] = {k: v for k, v in row.items() if v not in (None, "")}
    if "value" in data:
        data["value"] = _coerce_number(data["value"])
    if "sqi" in data:
        data["sqi"] = _coerce_number(data["sqi"])
    return data


def rows_from_text(text: str) -> Iterator[dict[str, Any]]:
    for row in csv.DictReader(io.StringIO(text)):
        yield _clean(row)


def rows_from_csv(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            yield _clean(row)
