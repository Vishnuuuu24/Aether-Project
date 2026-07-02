"""PatientStateEngine — validate + commit engine outputs into the versioned PSG,
audit every mutation, and serve the consent-scoped projection (docs/04 §3, §5, §7).

Commit rules:
  - Consent is re-checked at commit (deny-by-default; docs/02 §2). No VITALS consent
    => ConsentError, nothing is written.
  - Baselines are upserted as NEW versions only on material change (center/dispersion/
    fallback/method). A population-fallback → personalised switch is therefore a new,
    audited version — never a silent transition (docs/05 §8).
  - A deviation node is committed only when the reading actually deviates
    (magnitude != NORMAL); a normal reading is not a state change. Abstention is a
    correct outcome (CLAUDE.md).
  - An UNAVAILABLE baseline (no basis) is not persisted, and no deviation is written.
  - Every committed node emits an audit event stamped with the active versions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from core.audit import AuditWriter
from core.auth.consent_gate import granted_scopes, require_consent
from core.auth.errors import ConsentError
from core.versioning import VersionSet
from schemas.audit import AuditAction, AuditActor
from schemas.baseline import Baseline, BaselineAvailability, DeviationMagnitude, DeviationResult
from schemas.consent import ConsentScope
from schemas.patient import PatientProfile
from schemas.psg import BaselineNode, DeviationNode, PSGProjection

from .consent import ConsentProvider
from .profile import ProfileProvider
from .projection import build_projection
from .store import PSGStore

ACTOR_NAME = "patient-state-engine"
_FLOAT_REL_TOL = 1e-6
_FLOAT_ABS_TOL = 1e-9


class ProfileNotFoundError(LookupError):
    """No patient profile on file — the projection cannot be built."""


@dataclass(frozen=True)
class StateCommit:
    """Outcome of committing one scored reading."""

    baseline_node: BaselineNode | None
    baseline_committed: bool  # a new baseline version was written
    deviation_node: DeviationNode | None
    transition: bool  # population-fallback <-> personalised flip was recorded


class PatientStateEngine:
    def __init__(
        self,
        *,
        store: PSGStore,
        consent_provider: ConsentProvider,
        audit_writer: AuditWriter,
        versions: VersionSet,
        profile_provider: ProfileProvider | None = None,
        actor: AuditActor = AuditActor.SYSTEM,
        clock: Callable[[], datetime] | None = None,
        deviation_limit: int = 20,
    ) -> None:
        self._store = store
        self._consent = consent_provider
        self._audit = audit_writer
        self._versions = versions
        self._profiles = profile_provider
        self._actor = actor
        self._clock = clock or (lambda: datetime.now(UTC))
        self._deviation_limit = deviation_limit

    # -- commit --------------------------------------------------------------

    def commit_deviation(
        self, baseline: Baseline, deviation: DeviationResult, *, occurred_at: datetime
    ) -> StateCommit:
        """Validate consent, then commit the baseline (if changed) and the deviation
        (if it actually deviates). Raises ConsentError if VITALS is not consented.
        """
        if occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        if deviation.patient_id != baseline.patient_id:
            raise ValueError("deviation and baseline patient_id disagree")
        # Guard against mislinking a deviation to a baseline of a different series,
        # which would corrupt the PSG (deviation.baseline_id -> wrong metric/context).
        if deviation.metric_code != baseline.metric_code or deviation.context != baseline.context:
            raise ValueError("deviation and baseline metric_code/context disagree")

        consent = self._consent.get_consent(baseline.patient_id)
        require_consent(consent, ConsentScope.VITALS, patient_id=baseline.patient_id)

        baseline_node, baseline_committed, transition = self._commit_baseline(baseline, occurred_at)

        deviation_node: DeviationNode | None = None
        if baseline_node is not None and deviation.magnitude is not DeviationMagnitude.NORMAL:
            deviation_node = self._commit_deviation_node(deviation, baseline_node, occurred_at)

        return StateCommit(
            baseline_node=baseline_node,
            baseline_committed=baseline_committed,
            deviation_node=deviation_node,
            transition=transition,
        )

    def _commit_baseline(
        self, baseline: Baseline, occurred_at: datetime
    ) -> tuple[BaselineNode | None, bool, bool]:
        if baseline.availability is BaselineAvailability.UNAVAILABLE or baseline.center is None:
            return None, False, False
        assert baseline.dispersion_sigma is not None  # center set => sigma set

        current = self._store.current_baseline(
            baseline.patient_id, baseline.metric_code.value, baseline.context.value
        )
        if current is not None and not _baseline_changed(current, baseline):
            return current, False, False

        transition = (
            current is not None
            and current.is_population_fallback != baseline.is_population_fallback
        )
        node = BaselineNode(
            patient_id=baseline.patient_id,
            version=(current.version + 1) if current is not None else 1,
            supersedes=current.id if current is not None else None,
            created_at=occurred_at,
            created_by=ACTOR_NAME,
            metric_code=baseline.metric_code,
            context=baseline.context,
            method=baseline.method,
            center=baseline.center,
            dispersion=baseline.dispersion_sigma,
            sample_n=baseline.sample_n,
            window_spec=f"{baseline.window_days}d",
            confidence=_baseline_confidence(baseline),
            is_population_fallback=baseline.is_population_fallback,
        )
        self._store.add_baseline(node)

        input_refs = []
        if transition:
            assert current is not None
            src = "population_fallback" if current.is_population_fallback else "personalised"
            dst = "population_fallback" if baseline.is_population_fallback else "personalised"
            input_refs.append(f"transition:{src}->{dst}")
        self._write_audit(
            baseline.patient_id,
            AuditAction.BASELINE_UPDATE,
            input_refs=input_refs,
            output_refs=[f"baseline:{node.id}"],
        )
        return node, True, transition

    def _commit_deviation_node(
        self, deviation: DeviationResult, baseline_node: BaselineNode, occurred_at: datetime
    ) -> DeviationNode:
        node = DeviationNode(
            patient_id=deviation.patient_id,
            version=1,
            supersedes=None,
            created_at=occurred_at,
            created_by=ACTOR_NAME,
            metric_code=deviation.metric_code,
            baseline_id=baseline_node.id,
            magnitude=abs(deviation.z_robust),
            direction=deviation.direction,
            z_robust=deviation.z_robust,
            confidence=deviation.confidence,
            is_population_fallback=deviation.is_population_fallback,
        )
        self._store.add_deviation(node)
        self._write_audit(
            deviation.patient_id,
            AuditAction.STATE_COMMIT,
            input_refs=[f"reading:{deviation.reading_id}", f"baseline:{baseline_node.id}"],
            output_refs=[f"deviation:{node.id}"],
        )
        return node

    def _write_audit(
        self,
        patient_id: UUID,
        action: AuditAction,
        *,
        input_refs: list[str],
        output_refs: list[str],
    ) -> None:
        self._audit.write(
            patient_id=patient_id,
            actor=self._actor,
            action=action,
            input_refs=input_refs,
            output_refs=output_refs,
            versions=self._versions.as_dict(),
        )

    # -- read ----------------------------------------------------------------

    def build_projection(self, patient_id: UUID) -> PSGProjection:
        """Consent-scoped projection for `patient_id`. Raises ConsentError when the
        patient has no scopes in force, ProfileNotFoundError when no profile exists.
        """
        profile = self._get_profile(patient_id)
        consent = self._consent.get_consent(patient_id)
        # Deny-by-default: at least one scope must be in force to disclose anything.
        if not granted_scopes(consent):
            raise ConsentError(
                "consent gate denied: no consent scopes in force", patient_id=patient_id
            )
        return build_projection(
            patient_id=patient_id,
            store=self._store,
            consent=consent,
            profile=profile,
            versions=self._versions,
            now=self._clock(),
            deviation_limit=self._deviation_limit,
        )

    def _get_profile(self, patient_id: UUID) -> PatientProfile:
        if self._profiles is None:
            raise ProfileNotFoundError(str(patient_id))
        profile = self._profiles.get_profile(patient_id)
        if profile is None:
            raise ProfileNotFoundError(str(patient_id))
        return profile


def _baseline_confidence(baseline: Baseline) -> float:
    """Sufficiency-based reliability heuristic (uncalibrated). Not a clinical value:
    it scales with how much data backs the baseline and penalises the population
    fallback (docs/05 §5 — confidence is not shipped as calibrated in v1).
    """
    sample_factor = min(1.0, baseline.sample_n / baseline.min_n) if baseline.min_n else 1.0
    fallback_factor = 0.5 if baseline.is_population_fallback else 1.0
    return max(0.0, min(1.0, sample_factor * fallback_factor))


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=_FLOAT_REL_TOL, abs_tol=_FLOAT_ABS_TOL)


def _baseline_changed(current: BaselineNode, candidate: Baseline) -> bool:
    """A new version is warranted only on material change. Confidence is included so
    the stored reliability tracks growing evidence (sample_n) rather than going stale;
    once personalised it saturates, so this does not churn versions per reading.
    """
    assert candidate.center is not None and candidate.dispersion_sigma is not None
    return (
        current.is_population_fallback != candidate.is_population_fallback
        or current.method != candidate.method
        or not _same(current.center, candidate.center)
        or not _same(current.dispersion, candidate.dispersion_sigma)
        or not _same(current.confidence, _baseline_confidence(candidate))
    )
