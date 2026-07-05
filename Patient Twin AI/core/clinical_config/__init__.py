"""core.clinical_config — strict validation for the clinician-filled config stubs
(T8.3; docs/05, docs/06).

The loaders in `ai/features`, `ai/baseline`, `services/policy_engine`,
`services/event_engine` and `services/doc_coding_service` CONSUME these files
fail-safe (unset => inert). This package adds the missing VALIDATION layer: a gate
that fails loudly on a PRESENT-but-malformed value while leaving an all-unset stub
perfectly valid. It never fabricates clinical content (CLAUDE.md).
"""

from __future__ import annotations

from .errors import ClinicalConfigError
from .service import (
    DEFAULT_CLINICAL_CONFIG_DIR,
    ClinicalConfigReport,
    SectionReport,
    validate_clinical_configs,
)

__all__ = [
    "DEFAULT_CLINICAL_CONFIG_DIR",
    "ClinicalConfigError",
    "ClinicalConfigReport",
    "SectionReport",
    "validate_clinical_configs",
]
