"""Offline-dataset eval adapters (docs/11 §1.2; T8.2).

Turn on-disk clinical/wearable datasets into the harness-neutral
`LabelledDeviation`s that `ai/baseline/eval.py` scores — so the eval report
produces REAL precision/recall/F1 + ECE instead of synthetic smoke. Adapters
validate a dataset's layout before trusting its labels (CLAUDE.md: never fabricate
a benchmark from signals you haven't verified) and derive HR through the classical
`FeatureExtractor` (T8.1), never a bespoke path.
"""

from .wesad import (
    WESAD_STRESS_LABEL,
    WesadLayoutError,
    load_wesad_labelled_deviations,
    wesad_available,
)

__all__ = [
    "WESAD_STRESS_LABEL",
    "WesadLayoutError",
    "load_wesad_labelled_deviations",
    "wesad_available",
]
