"""Version registry — the active model / ruleset / prompt / baseline-engine /
schema versions stamped on every output (CLAUDE.md; docs/04, docs/06 §8).

Read-mostly by design. Versions change ONLY through human-gated releases
(`with_versions` returns a NEW registry; nothing mutates in place). There is no
path by which the LLM or any service rewrites these — that would be the forbidden
closed-loop self-modification (CLAUDE.md principle 5).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace

from schemas.output_contract import VersionStamp as OutputVersionStamp
from schemas.psg import VersionStamp as ProjectionVersionStamp

# Until a real release pins them, versions read from env with explicit fallbacks.
_UNSET = "unset"


@dataclass(frozen=True)
class VersionSet:
    model: str
    ruleset: str
    prompt: str
    baseline_engine: str
    schema: str

    def output_stamp(self) -> OutputVersionStamp:
        """Stamp for the user-facing output contract (docs/04 §6)."""
        return OutputVersionStamp(
            model=self.model,
            ruleset=self.ruleset,
            baseline_engine=self.baseline_engine,
            prompt=self.prompt,
        )

    def projection_stamp(self) -> ProjectionVersionStamp:
        """Stamp for the PSG projection handed to the LLM (docs/04 §5)."""
        return ProjectionVersionStamp(
            baseline_engine=self.baseline_engine,
            ruleset=self.ruleset,
            prompt=self.prompt,
            model=self.model,
        )

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class VersionRegistry:
    def __init__(self, versions: VersionSet) -> None:
        self._versions = versions

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> VersionRegistry:
        env = env if env is not None else os.environ
        return cls(
            VersionSet(
                model=env.get("PRIMARY_MODEL", _UNSET),
                ruleset=env.get("RULESET_VERSION", _UNSET),
                prompt=env.get("PROMPT_VERSION", _UNSET),
                baseline_engine=env.get("BASELINE_ENGINE_VERSION", _UNSET),
                schema=env.get("SCHEMA_VERSION", _UNSET),
            )
        )

    def current(self) -> VersionSet:
        return self._versions

    def output_stamp(self) -> OutputVersionStamp:
        return self._versions.output_stamp()

    def projection_stamp(self) -> ProjectionVersionStamp:
        return self._versions.projection_stamp()

    def with_versions(self, **overrides: str) -> VersionRegistry:
        """Return a NEW registry with some versions changed. Models a human-gated
        release — never mutates the existing registry in place.
        """
        unknown = set(overrides) - set(self._versions.as_dict())
        if unknown:
            raise ValueError(f"unknown version fields: {sorted(unknown)}")
        return VersionRegistry(replace(self._versions, **overrides))
