"""The single error type raised when a clinical config is malformed (T8.3).

UNSET is never an error — the whole design is fail-safe on unset (CLAUDE.md). This
is raised only when a value is PRESENT but wrong (bad type, out-of-range, invalid
enum, or a partially-filled row), so a clinician's config mistake surfaces loudly
instead of silently degrading to the fail-safe default.
"""

from __future__ import annotations


class ClinicalConfigError(ValueError):
    """A present-but-invalid entry in a clinical config file."""

    def __init__(self, section: str, reason: str, *, key: str | None = None) -> None:
        self.section = section
        self.key = key
        self.reason = reason
        where = f"{section}[{key}]" if key is not None else section
        super().__init__(f"clinical config '{where}': {reason}")
