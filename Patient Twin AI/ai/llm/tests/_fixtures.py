"""Deterministic fixtures for gateway tests: a minimal projection, evidence, and a
fake `ChatClient` that returns a canned JSON string (no live model needed)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from schemas.psg import (
    BaselineSummary,
    DeviationDirection,
    DeviationSummary,
    PSGProjection,
    VersionStamp,
)
from schemas.reading import MeasurementContext, MetricCode
from schemas.retrieval import EvidenceChunk
from schemas.vector import VectorSourceType

AS_OF = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
KB_CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")


def make_projection(*, population_fallback: bool = False) -> PSGProjection:
    return PSGProjection(
        patient_age_years=41,
        patient_sex_at_birth="female",
        baselines=[
            BaselineSummary(
                metric_code=MetricCode.HEART_RATE,
                context=MeasurementContext.RESTING,
                center=58.0,
                dispersion=4.0,
                confidence=0.9,
                is_population_fallback=population_fallback,
            )
        ],
        recent_deviations=[
            DeviationSummary(
                metric_code=MetricCode.HEART_RATE,
                direction=DeviationDirection.UP,
                magnitude=12.0,
                z_robust=3.1,
                confidence=0.8,
                ts=AS_OF,
            )
        ],
        as_of=AS_OF,
        consent_scope=["copilot", "vitals"],
        versions=VersionStamp(baseline_engine="stat-v1", ruleset="unset", prompt="p1", model="m1"),
    )


def make_evidence() -> list[EvidenceChunk]:
    return [
        EvidenceChunk(
            chunk_id=KB_CHUNK_ID,
            source_type=VectorSourceType.KB_PASSAGE,
            text="Resting heart rate rises transiently with acute stress or illness.",
            score=0.9,
            section="cardiology",
        )
    ]


class FakeChatClient:
    """Records the last call and returns a preset response (or raises a preset exc)."""

    def __init__(self, response: str | None = None, raises: Exception | None = None) -> None:
        self._response = response
        self._raises = raises
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        schema_name: str,
        temperature: float,
    ) -> str:
        self.last_system = system
        self.last_user = user
        if self._raises is not None:
            raise self._raises
        assert self._response is not None
        return self._response
