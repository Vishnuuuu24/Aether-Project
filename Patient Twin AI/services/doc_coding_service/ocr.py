"""OCR adapters (docs/03 §2.6: Docling + Marker, text documents only in v1).

`PassthroughOcr` is the dev/test engine for already-textual documents. `DoclingOcr`
is the real adapter; it imports Docling lazily so the base install and fast test
suite need no heavy ML dependency.
"""

from __future__ import annotations

from schemas.document import ClinicalDocument


class PassthroughOcr:
    """Returns the document's inline text. For dev and already-OCR'd inputs."""

    def extract_text(self, document: ClinicalDocument) -> str:
        if document.text and document.text.strip():
            return document.text
        raise ValueError("PassthroughOcr requires inline document.text")


class DoclingOcr:
    """Real OCR via Docling (docs/03 §2.6). Lazily imports `docling`; converts the
    document `uri` to text/markdown. Raises ImportError if docling is not installed
    (integration tests skip in that case).
    """

    def __init__(self) -> None:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - exercised only with docling installed
            raise ImportError(
                "docling is not installed; install requirements-ml.txt to use DoclingOcr"
            ) from exc
        self._converter = DocumentConverter()

    def extract_text(self, document: ClinicalDocument) -> str:  # pragma: no cover - needs docling
        if not document.uri:
            raise ValueError("DoclingOcr requires document.uri")
        result = self._converter.convert(document.uri)
        return str(result.document.export_to_markdown())
