"""Document & Coding Service (docs/04 §4; T3.1).

Pipeline: ClinicalDocument -> OCR -> clinical coding -> confidence-gated CodedEntity[]
(sub-threshold stays `proposed`, never silently committed). The Patient State Engine
commits the results onto the PSG's Condition/Medication/Observation/Allergy nodes.

Model backends are service-local ports (OcrEngine, ClinicalCoder). Real adapters:
DoclingOcr (OCR) and MedCatCoder (SNOMED/RxNorm/LOINC) load their libraries/weights
lazily; the dev DictionaryCoder + PassthroughOcr drive the deterministic tests.
"""

from .coder import DictionaryCoder, MedCatCoder, TermCode
from .config import DEFAULT_CODING_THRESHOLDS_PATH, load_coding_thresholds
from .ocr import DoclingOcr, PassthroughOcr
from .ports import ClinicalCoder, OcrEngine
from .service import DocCodingService

__all__ = [
    "DEFAULT_CODING_THRESHOLDS_PATH",
    "ClinicalCoder",
    "DictionaryCoder",
    "DocCodingService",
    "DoclingOcr",
    "MedCatCoder",
    "OcrEngine",
    "PassthroughOcr",
    "TermCode",
    "load_coding_thresholds",
]
