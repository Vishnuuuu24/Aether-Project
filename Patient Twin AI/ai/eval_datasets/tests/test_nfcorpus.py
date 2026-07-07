"""Sprint 11: nfcorpus IR adapter — layout validation on synthetic parquet/qrels
(CI-safe) + a skip-guarded sanity run on the real dataset when present on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.eval_datasets.nfcorpus import (
    NfcorpusLayoutError,
    load_nfcorpus_split,
    nfcorpus_available,
)

_REAL_ROOT = Path("datasets/nfcorpus")


def _write_parquet(path: Path, rows: list[dict[str, str]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "_id": [r["_id"] for r in rows],
            "title": [r["title"] for r in rows],
            "text": [r["text"] for r in rows],
        }
    )
    pq.write_table(table, path)


def _write_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["query-id\tcorpus-id\tscore"]
    lines += [f"{q}\t{d}\t{s}" for q, d, s in rows]
    path.write_text("\n".join(lines) + "\n")


def _build_corpus(root: Path) -> None:
    _write_parquet(
        root / "corpus" / "corpus-00000-of-00001.parquet",
        [
            {"_id": "MED-1", "title": "Statins", "text": "cholesterol lowering drugs"},
            {"_id": "MED-2", "title": "Aspirin", "text": "antiplatelet therapy"},
            {"_id": "MED-3", "title": "", "text": "no title body only"},
        ],
    )
    _write_parquet(
        root / "queries" / "queries-00000-of-00001.parquet",
        [
            {"_id": "PLAIN-1", "title": "", "text": "do statins lower cholesterol"},
            {"_id": "PLAIN-2", "title": "", "text": "aspirin for heart"},
        ],
    )


pytest.importorskip("pyarrow")


def test_load_split_parses_and_joins(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    _write_qrels(
        tmp_path / "qrels" / "test.tsv",
        [("PLAIN-1", "MED-1", 2), ("PLAIN-2", "MED-2", 1)],
    )
    split = load_nfcorpus_split("test", tmp_path)
    assert split.split == "test"
    assert set(split.query_ids) == {"PLAIN-1", "PLAIN-2"}
    # title + body get joined; title-less doc keeps the body only.
    assert split.corpus["MED-1"] == "Statins. cholesterol lowering drugs"
    assert split.corpus["MED-3"] == "no title body only"
    assert split.relevant_ids("PLAIN-1") == frozenset({"MED-1"})


def test_unknown_split_raises(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    _write_qrels(tmp_path / "qrels" / "test.tsv", [("PLAIN-1", "MED-1", 1)])
    with pytest.raises(NfcorpusLayoutError, match="unknown split"):
        load_nfcorpus_split("holdout", tmp_path)


def test_qrels_pointing_at_unknown_doc_raises(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    _write_qrels(tmp_path / "qrels" / "test.tsv", [("PLAIN-1", "MED-999", 1)])
    with pytest.raises(NfcorpusLayoutError, match="absent from corpus"):
        load_nfcorpus_split("test", tmp_path)


def test_qrels_pointing_at_unknown_query_raises(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    _write_qrels(tmp_path / "qrels" / "test.tsv", [("PLAIN-999", "MED-1", 1)])
    with pytest.raises(NfcorpusLayoutError, match="absent from queries"):
        load_nfcorpus_split("test", tmp_path)


def test_bad_qrels_header_raises(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    (tmp_path / "qrels").mkdir(parents=True, exist_ok=True)
    (tmp_path / "qrels" / "test.tsv").write_text("q\td\ts\nPLAIN-1\tMED-1\t1\n")
    with pytest.raises(NfcorpusLayoutError, match="unexpected qrels header"):
        load_nfcorpus_split("test", tmp_path)


def test_query_with_only_zero_scores_is_excluded(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    _write_qrels(
        tmp_path / "qrels" / "test.tsv",
        [("PLAIN-1", "MED-1", 0), ("PLAIN-2", "MED-2", 1)],
    )
    split = load_nfcorpus_split("test", tmp_path)
    assert set(split.query_ids) == {"PLAIN-2"}  # PLAIN-1 had no relevant doc


@pytest.mark.skipif(not nfcorpus_available(_REAL_ROOT), reason="real nfcorpus not on disk")
def test_real_nfcorpus_sanity() -> None:
    split = load_nfcorpus_split("test", _REAL_ROOT)
    assert len(split.queries) > 100
    assert len(split.corpus) > 3000
    # every included query has at least one relevant doc, all present in the corpus
    for qid in split.query_ids:
        rel = split.relevant_ids(qid)
        assert rel
        assert rel <= set(split.corpus)
