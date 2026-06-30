"""Version registry: env loading, output/projection stamps, and immutability."""

from __future__ import annotations

import pytest

from core.versioning import VersionRegistry

ENV = {
    "PRIMARY_MODEL": "Qwen/Qwen3.6-35B-A3B",
    "RULESET_VERSION": "2026.06.0",
    "PROMPT_VERSION": "prompt-1",
    "BASELINE_ENGINE_VERSION": "stat-baseline-1",
    "SCHEMA_VERSION": "schema-1",
}


def test_from_env_reads_all_fields() -> None:
    reg = VersionRegistry.from_env(ENV)
    cur = reg.current()
    assert cur.model == "Qwen/Qwen3.6-35B-A3B"
    assert cur.ruleset == "2026.06.0"
    assert cur.prompt == "prompt-1"
    assert cur.baseline_engine == "stat-baseline-1"
    assert cur.schema == "schema-1"


def test_from_env_defaults_to_unset() -> None:
    reg = VersionRegistry.from_env({})
    assert reg.current().model == "unset"


def test_output_and_projection_stamps() -> None:
    reg = VersionRegistry.from_env(ENV)
    out = reg.output_stamp()
    assert (out.model, out.ruleset, out.baseline_engine, out.prompt) == (
        "Qwen/Qwen3.6-35B-A3B",
        "2026.06.0",
        "stat-baseline-1",
        "prompt-1",
    )
    proj = reg.projection_stamp()
    assert proj.model == "Qwen/Qwen3.6-35B-A3B"
    assert proj.baseline_engine == "stat-baseline-1"


def test_with_versions_is_immutable() -> None:
    reg = VersionRegistry.from_env(ENV)
    bumped = reg.with_versions(ruleset="2026.07.0")
    assert reg.current().ruleset == "2026.06.0"  # original untouched
    assert bumped.current().ruleset == "2026.07.0"


def test_with_versions_rejects_unknown_field() -> None:
    reg = VersionRegistry.from_env(ENV)
    with pytest.raises(ValueError):
        reg.with_versions(bogus="x")
