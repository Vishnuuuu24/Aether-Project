"""Promotion *recommendation* — advisory, non-mutating (docs/16 Sprint 10; CLAUDE.md §5).

A trained checkpoint that beats its bars is *recommended* for promotion, but promotion
itself — swapping the live version + emitting the audit event — is a **human-gated**
action (no closed-loop self-modification). This module only produces an advisory record:
it compares a checkpoint's measured metrics against explicit bars and writes a
`promotion.json` next to the artifact. It never edits the `VersionRegistry`, routing,
rulesets, or any live config; a human reads the recommendation and decides.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Bar:
    """One promotion criterion: `value` must beat `threshold` in `direction`."""

    name: str
    value: float
    threshold: float
    higher_is_better: bool

    @property
    def passed(self) -> bool:
        if self.higher_is_better:
            return self.value >= self.threshold
        return self.value <= self.threshold


@dataclass(frozen=True)
class PromotionRecommendation:
    """Advisory outcome — NOT a promotion. `recommended` is true only if every bar passes."""

    version: str
    recommended: bool
    bars: tuple[Bar, ...]
    rationale: str
    decided_by: str = "automated-eval (advisory only; human gate required)"

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "recommended": self.recommended,
            "decided_by": self.decided_by,
            "rationale": self.rationale,
            "bars": [asdict(b) | {"passed": b.passed} for b in self.bars],
        }


def evaluate_promotion(version: str, bars: list[Bar]) -> PromotionRecommendation:
    """Build a recommendation: recommended iff ALL bars pass. Pure; no side effects."""
    passed = [b for b in bars if b.passed]
    recommended = len(passed) == len(bars) and bool(bars)
    if recommended:
        rationale = f"beats all {len(bars)} bar(s): " + ", ".join(
            f"{b.name}={b.value:.3f} vs {b.threshold:.3f}" for b in bars
        )
    else:
        failed = [b for b in bars if not b.passed]
        rationale = "does not beat: " + ", ".join(
            f"{b.name}={b.value:.3f} vs {b.threshold:.3f}" for b in failed
        )
    return PromotionRecommendation(
        version=version, recommended=recommended, bars=tuple(bars), rationale=rationale
    )


def write_promotion_recommendation(
    recommendation: PromotionRecommendation, checkpoint_dir: Path
) -> Path:
    """Write `promotion.json` into the checkpoint dir. Advisory artifact only — this
    does NOT promote the model (no registry/routing mutation; CLAUDE.md §5)."""
    path = checkpoint_dir / "promotion.json"
    path.write_text(json.dumps(recommendation.to_dict(), indent=2))
    return path
