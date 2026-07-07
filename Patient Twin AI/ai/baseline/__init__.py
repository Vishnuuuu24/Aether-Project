"""v1 statistical baseline & deviation engine (docs/05).

`StatisticalBaselineEngine` implements the stable `BaselineEngine` interface:
rolling median/MAD center + dispersion, EWMA trend, opportunistic circadian
stratification, sufficiency-gated personalisation, and a labelled population
fallback. Deviation scoring produces `DeviationResult`s (docs/05 §5).
"""

from .config import BaselineConfig, circadian_bucket
from .foundation_encoder import (
    FOUNDATION_ENCODER_BASELINE_VERSION,
    FoundationEncoderBaselineEngine,
)
from .population import (
    PopulationReferenceProvider,
    StaticPopulationReferenceProvider,
    YamlPopulationReferenceProvider,
)
from .statistical import BASELINE_ENGINE_VERSION, StatisticalBaselineEngine

__all__ = [
    "BASELINE_ENGINE_VERSION",
    "FOUNDATION_ENCODER_BASELINE_VERSION",
    "BaselineConfig",
    "FoundationEncoderBaselineEngine",
    "PopulationReferenceProvider",
    "StaticPopulationReferenceProvider",
    "StatisticalBaselineEngine",
    "YamlPopulationReferenceProvider",
    "circadian_bucket",
]
