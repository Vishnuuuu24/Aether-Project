"""De-identification egress filter (docs/06 §6, §9).

A conservative, DEFAULT-DENY guard that sits in front of any `external_*` profile.
It scans the exact bytes about to leave the trust boundary (the assembled prompt)
for direct identifiers. If anything fires — or if the scan cannot be run — egress is
BLOCKED. This is deliberately biased toward false positives: blocking a clean payload
is a nuisance; leaking PHI to a third party is not recoverable.

This is NOT a de-identification *transformer* — it does not scrub and forward. In v1
it is a gate: clean payloads pass, anything suspicious is refused and the copilot
abstains (docs/06 §9 "De-identification uncertain → block external egress"). Real
production traffic never reaches here because it is hard-pinned to the `local`
profile, which is PHI-allowed and skips this filter entirely.

The patterns below catch the common direct identifiers (HIPAA Safe Harbor-style).
They are intentionally broad; a clinical/privacy review tightens them before any
external profile is used with anything approaching real data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class EgressBlocked(Exception):
    """Raised when a payload bound for an external profile is not provably clean."""

    def __init__(self, kinds: list[str]) -> None:
        self.kinds = kinds
        super().__init__(f"egress blocked — possible identifiers detected: {sorted(set(kinds))}")


# (kind, compiled pattern). Broad on purpose (default-deny).
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # NANP-ish / international phone numbers with separators.
    (
        "phone",
        re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"),
    ),
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    # A full calendar date (DOB / admission date) — day-level dates are identifiers.
    ("date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    (
        "date",
        re.compile(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ),
    ("url", re.compile(r"https?://\S+")),
    ("ip", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    # Medical record / account numbers: a token that is a long-ish digit run.
    ("record_number", re.compile(r"(?<!\d)\d{7,}(?!\d)")),
    # US-style street address ("123 Main St").
    (
        "address",
        re.compile(
            r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
            r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b"
        ),
    ),
]


@dataclass
class DeidReport:
    clean: bool
    matches: dict[str, list[str]] = field(default_factory=dict)

    @property
    def kinds(self) -> list[str]:
        return list(self.matches)


def scan_for_identifiers(text: str) -> DeidReport:
    """Report any direct identifiers found. Empty match set => provisionally clean.

    Note: "clean" here means "no configured pattern fired", NOT "provably free of
    PHI". Names and free-text quasi-identifiers are not detectable by regex; this is
    why external profiles are dev-only and why production stays on `local`.
    """
    matches: dict[str, list[str]] = {}
    for kind, pattern in _PATTERNS:
        found = pattern.findall(text)
        if found:
            matches.setdefault(kind, []).extend(m if isinstance(m, str) else str(m) for m in found)
    return DeidReport(clean=not matches, matches=matches)


def assert_clean_for_egress(text: str) -> None:
    """Default-deny gate. Raises `EgressBlocked` if anything looks like an identifier."""
    report = scan_for_identifiers(text)
    if not report.clean:
        raise EgressBlocked(report.kinds)
