"""Shared adapter types.

A `CanonicalReading` is a dict using the Reading field names (docs/04 §2). Adapters
emit these; the normaliser fills defaults (sqi unknown, ingest_adapter,
included_in_baseline=False) and validates. Missing fields are the normaliser's job
to reject — adapters don't pre-validate.
"""

from __future__ import annotations

from typing import Any

CanonicalReading = dict[str, Any]
