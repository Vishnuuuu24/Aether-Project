"""Coding-threshold stub loader + real-adapter availability (skip/deferred)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from services.doc_coding_service.coder import MedCatCoder
from services.doc_coding_service.config import (
    DEFAULT_CODING_THRESHOLDS_PATH,
    load_coding_thresholds,
)


def test_shipped_stub_is_all_unset() -> None:
    # The committed clinical stub must carry no fabricated thresholds.
    assert load_coding_thresholds(DEFAULT_CODING_THRESHOLDS_PATH) == {}


def test_missing_file_returns_empty() -> None:
    assert load_coding_thresholds(Path("config/clinical/nope.yaml")) == {}


def test_loads_only_set_values(tmp_path: Path) -> None:
    cfg = tmp_path / "coding.yaml"
    cfg.write_text("thresholds:\n  condition: 0.7\n  medication:\n")
    assert load_coding_thresholds(cfg) == {"condition": 0.7}


def test_medcat_coder_deferred_without_license() -> None:
    # Real MedCAT is deferred until a UMLS-licensed model pack is provided.
    if os.environ.get("MEDCAT_MODEL_PACK"):
        pytest.skip("MEDCAT_MODEL_PACK is set — real MedCAT path exercised elsewhere")
    with pytest.raises(RuntimeError, match="MEDCAT_MODEL_PACK"):
        MedCatCoder()


def test_docling_ocr_skips_when_uninstalled() -> None:
    pytest.importorskip("docling", reason="docling not installed — real OCR adapter deferred")
    from services.doc_coding_service.ocr import DoclingOcr

    DoclingOcr()  # constructs only when docling is present
