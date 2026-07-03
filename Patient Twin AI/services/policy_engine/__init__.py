"""Deterministic Policy Engine (docs/06, docs/10 T4.3).

The last gate before any user-facing output. It takes an LLM `ProposedOutput` and the
patient's `PSGProjection` + retrieved evidence, runs ordered deterministic checks
(first hard failure wins), and produces the final `OutputContract`. It is the ONLY
component allowed to emit an OutputContract with `decision == approved`; it can also
downgrade, suppress, or force-abstain. It is versioned and never edited by the LLM
(CLAUDE.md principles 4 & 5).
"""

from .engine import POLICY_ENGINE_VERSION, PolicyEngine
from .rules import PolicyRuleSet, RedFlagRule, load_policy_rules

__all__ = [
    "POLICY_ENGINE_VERSION",
    "PolicyEngine",
    "PolicyRuleSet",
    "RedFlagRule",
    "load_policy_rules",
]
