"""v1 classical feature extraction + SQI quality gate (docs/05 §3-4).

`ClassicalFeatureExtractor` implements the stable `FeatureExtractor` interface;
`SqiGate` enforces the per-metric quality precondition before a reading may enter
the personal baseline.
"""

from .classical import FEATURE_EXTRACTOR_VERSION, ClassicalFeatureExtractor
from .config import DEFAULT_SQI_THRESHOLDS_PATH, load_sqi_thresholds
from .sqi import SqiGate
from .waveform import HrHrvResult, extract_hr_hrv
from .waveform_extractor import (
    FEATURE_EXTRACTOR_VERSION as WAVEFORM_FEATURE_EXTRACTOR_VERSION,
)
from .waveform_extractor import WaveformFeatureExtractor

__all__ = [
    "DEFAULT_SQI_THRESHOLDS_PATH",
    "FEATURE_EXTRACTOR_VERSION",
    "WAVEFORM_FEATURE_EXTRACTOR_VERSION",
    "ClassicalFeatureExtractor",
    "HrHrvResult",
    "SqiGate",
    "WaveformFeatureExtractor",
    "extract_hr_hrv",
    "load_sqi_thresholds",
]
