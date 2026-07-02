"""HybridRetriever — retrieval quality + T3.2 DoD: consent-scoped retrieval."""

from __future__ import annotations

from uuid import uuid4

from ai.retrieval.embedder import HashEmbedder
from ai.retrieval.hybrid import HybridRetriever
from ai.retrieval.reranker import LexicalReranker
from ai.retrieval.vector_store import InMemoryVectorStore
from schemas.consent import ConsentScope
from schemas.retrieval import RetrievalScope
from schemas.vector import VectorPayload, VectorSourceType

from ._corpus import kb, patient_chunk


def _retriever(corpus: list[VectorPayload]) -> HybridRetriever:
    return HybridRetriever(
        corpus,
        embedder=HashEmbedder(),
        reranker=LexicalReranker(),
        vector_store=InMemoryVectorStore(),
    )


def test_retrieves_topically_relevant_passage() -> None:
    corpus = [
        kb("metformin is first-line therapy for type 2 diabetes", index=0),
        kb("atrial fibrillation requires anticoagulation", index=1),
    ]
    results = _retriever(corpus).search(
        "type 2 diabetes treatment", RetrievalScope(include_kb=True), k=2
    )
    assert results
    assert "diabetes" in results[0].text


def test_patient_records_scoped_to_owner() -> None:
    a, b = uuid4(), uuid4()
    corpus = [
        kb("general blood pressure guidance"),
        patient_chunk(a, "patient reports elevated blood pressure at home"),
        patient_chunk(b, "other patient elevated blood pressure log"),
    ]
    results = _retriever(corpus).search(
        "blood pressure",
        RetrievalScope(patient_id=a, consented_scopes=[ConsentScope.DOCUMENTS], include_kb=True),
        k=10,
    )
    patient_ids = {c.patient_id for c in results if c.patient_id is not None}
    assert a in patient_ids  # own records surface
    assert b not in patient_ids  # another patient's records never surface


def test_patient_records_hidden_without_consented_scope() -> None:
    a = uuid4()
    corpus = [kb("general blood pressure guidance"), patient_chunk(a, "elevated blood pressure")]
    results = _retriever(corpus).search(
        "blood pressure",
        RetrievalScope(patient_id=a, consented_scopes=[], include_kb=True),  # no DOCUMENTS scope
        k=10,
    )
    assert all(c.source_type is VectorSourceType.KB_PASSAGE for c in results)


def test_kb_excluded_when_not_requested() -> None:
    a = uuid4()
    corpus = [kb("general blood pressure guidance"), patient_chunk(a, "elevated blood pressure")]
    results = _retriever(corpus).search(
        "blood pressure",
        RetrievalScope(patient_id=a, consented_scopes=[ConsentScope.DOCUMENTS], include_kb=False),
        k=10,
    )
    assert results
    assert all(c.source_type is VectorSourceType.PATIENT_RECORD for c in results)


def test_empty_corpus_returns_nothing() -> None:
    assert _retriever([]).search("anything", RetrievalScope(include_kb=True), k=5) == []
