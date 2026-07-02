"""Clinical coding adapters (docs/03 §2.6: MedCAT -> SNOMED CT / LOINC / RxNorm).

`MedCatCoder` is the real adapter; it needs a MedCAT model pack (a UMLS/SNOMED-
licensed artefact — a credential, never fabricated or committed) supplied via
`MEDCAT_MODEL_PACK`, and it imports MedCAT lazily. Until that license/pack is
provided, its integration test skips.

`DictionaryCoder` is a labelled DEV coder for tests/harness: it emits codes ONLY
from a caller-supplied phrase→code map. It ships with no clinical mappings — no
fabricated codes live in the repo (CLAUDE.md).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from schemas.document import CodedEntity, DocumentType, EntityType


@dataclass(frozen=True)
class TermCode:
    """A dev mapping target: what a matched phrase codes to."""

    entity_type: EntityType
    code_system: str
    code: str
    display: str
    confidence: float
    value: str | None = None
    unit: str | None = None


class DictionaryCoder:
    """DEV coder. Case-insensitive substring match against a supplied phrase map.
    Not clinical truth — the real coder is MedCAT.
    """

    def __init__(
        self, mappings: Mapping[str, TermCode], *, version: str = "dictionary-dev"
    ) -> None:
        self._mappings = dict(mappings)
        self.version = version

    def code(self, text: str, *, doc_type: DocumentType) -> list[CodedEntity]:
        haystack = text.lower()
        entities: list[CodedEntity] = []
        for phrase, tc in self._mappings.items():
            if phrase.lower() in haystack:
                entities.append(
                    CodedEntity(
                        entity_type=tc.entity_type,
                        code_system=tc.code_system,
                        code=tc.code,
                        display=tc.display,
                        confidence=tc.confidence,
                        value=tc.value,
                        unit=tc.unit,
                    )
                )
        return entities


class MedCatCoder:
    """Real MedCAT adapter (docs/03 §2.6). Loads a model pack from `MEDCAT_MODEL_PACK`
    and imports MedCAT lazily. Raises if MedCAT or the licensed pack is unavailable —
    integration tests skip until a UMLS-licensed pack is supplied.
    """

    _CUI_ENTITY_TYPE: dict[str, EntityType] = {}  # TDL: map TUI/semantic types -> EntityType

    def __init__(self, model_pack_path: str | None = None, *, version: str = "medcat") -> None:
        pack = model_pack_path or os.environ.get("MEDCAT_MODEL_PACK")
        if not pack:
            raise RuntimeError(
                "MEDCAT_MODEL_PACK is not set — provide a UMLS-licensed MedCAT model pack "
                "(a credential; never fabricated or committed) to enable real coding"
            )
        try:
            from medcat.cat import CAT
        except ImportError as exc:  # pragma: no cover - needs medcat installed
            raise ImportError(
                "medcat is not installed; install requirements-ml.txt to use MedCatCoder"
            ) from exc
        self._cat = CAT.load_model_pack(pack)  # pragma: no cover - needs licensed pack
        self.version = version

    def code(self, text: str, *, doc_type: DocumentType) -> list[CodedEntity]:  # pragma: no cover
        # Real mapping (MedCAT entities -> CodedEntity with SNOMED/RxNorm/LOINC + confidence)
        # lands with the licensed model pack; the pipeline + gate around it are tested via
        # DictionaryCoder so this adapter is a thin, weight-dependent shell.
        raise NotImplementedError(
            "MedCatCoder.code is wired to a licensed model pack; enable once MEDCAT_MODEL_PACK "
            "and the entity-type mapping are provisioned"
        )
