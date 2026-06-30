"""Audit integrity errors."""

from __future__ import annotations


class AuditChainError(Exception):
    """The hash chain is broken: a record's hash or prev_hash does not verify.

    `index` is the 0-based position of the first offending record.
    """

    def __init__(self, message: str, *, index: int | None = None) -> None:
        super().__init__(message)
        self.index = index
