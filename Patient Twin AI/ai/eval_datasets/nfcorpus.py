"""BEIR nfcorpus → medical IR eval/train adapter (docs/16 Sprint 11).

nfcorpus (Boteva et al., 2016) is a consumer-health information-retrieval benchmark
in the BEIR suite: ~3.6k PubMed documents, ~3.2k natural-language health queries, and
**human relevance judgements** (qrels) split into train/dev/test. We use it as the
labelled medical-IR signal for the Sprint 11 cross-encoder reranker — real query→passage
relevance from the qrels, never fabricated labels (CLAUDE.md).

Layout on disk (fetched by `scripts/fetch_datasets.py --dataset nfcorpus`):
    datasets/nfcorpus/corpus/corpus-*.parquet     (_id, title, text)
    datasets/nfcorpus/queries/queries-*.parquet   (_id, title, text)
    datasets/nfcorpus/qrels/{train,dev,test}.tsv  (query-id \t corpus-id \t score)

The qrels are graded (1 or 2). Downstream binary metrics treat score >= 1 as relevant;
the pair builder can weight by the graded score. Doc/query ids are BEIR strings
(`MED-10`, `PLAIN-3`) — kept as-is; nothing here invents a UUID mapping.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DEFAULT_NFCORPUS_ROOT = Path("datasets/nfcorpus")
_SPLITS = ("train", "dev", "test")
_MIN_RELEVANT_SCORE = 1  # qrels score >= this counts as relevant (nfcorpus grades 1,2)


class NfcorpusLayoutError(ValueError):
    """Raised when the nfcorpus layout is missing/malformed — so we never build an IR
    eval or training set from mismatched query/corpus/qrels files."""


@dataclass(frozen=True)
class IrSplit:
    """A labelled IR split: query/doc text plus graded relevance judgements.

    `qrels[qid]` maps a relevant doc-id to its graded score (1 or 2). Only queries that
    have at least one judged-relevant doc are included (unjudged queries carry no signal).
    """

    split: str
    queries: dict[str, str]  # qid -> query text
    corpus: dict[str, str]  # docid -> passage text (title + body)
    qrels: dict[str, dict[str, int]]  # qid -> {docid: score}

    @property
    def query_ids(self) -> tuple[str, ...]:
        return tuple(self.queries.keys())

    def relevant_ids(self, qid: str) -> frozenset[str]:
        return frozenset(d for d, s in self.qrels.get(qid, {}).items() if s >= _MIN_RELEVANT_SCORE)


def nfcorpus_available(root: Path = DEFAULT_NFCORPUS_ROOT) -> bool:
    """True iff corpus, queries, and all three qrels splits are present on disk."""
    if not (_parquet_files(root / "corpus") and _parquet_files(root / "queries")):
        return False
    return all((root / "qrels" / f"{s}.tsv").is_file() for s in _SPLITS)


def _parquet_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.parquet")) if directory.is_dir() else []


def _read_id_text(directory: Path, *, kind: str) -> dict[str, str]:
    """Read a BEIR parquet folder (_id/title/text) into `{id: "title. text"}`."""
    import pyarrow.parquet as pq

    files = _parquet_files(directory)
    if not files:
        raise NfcorpusLayoutError(f"no parquet under {directory} — cannot load nfcorpus {kind}")
    out: dict[str, str] = {}
    for path in files:
        table = pq.read_table(path, columns=["_id", "title", "text"])
        ids = table.column("_id").to_pylist()
        titles = table.column("title").to_pylist()
        texts = table.column("text").to_pylist()
        for _id, title, text in zip(ids, titles, texts, strict=True):
            title = (title or "").strip()
            text = (text or "").strip()
            out[str(_id)] = f"{title}. {text}" if title else text
    if not out:
        raise NfcorpusLayoutError(f"{directory} parsed to zero {kind} rows")
    return out


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse a BEIR qrels TSV (`query-id \t corpus-id \t score`, with header)."""
    if not path.is_file():
        raise NfcorpusLayoutError(f"missing qrels file {path}")
    qrels: dict[str, dict[str, int]] = {}
    lines = path.read_text().splitlines()
    if not lines:
        raise NfcorpusLayoutError(f"empty qrels file {path}")
    header = lines[0].split("\t")
    if header[:3] != ["query-id", "corpus-id", "score"]:
        raise NfcorpusLayoutError(f"unexpected qrels header in {path}: {lines[0]!r}")
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise NfcorpusLayoutError(f"{path}:{lineno} malformed qrels row: {line!r}")
        qid, docid, raw = parts[0], parts[1], parts[2]
        try:
            score = int(raw)
        except ValueError as exc:
            raise NfcorpusLayoutError(f"{path}:{lineno} non-integer score {raw!r}") from exc
        qrels.setdefault(qid, {})[docid] = score
    return qrels


def load_nfcorpus_split(split: str, root: Path = DEFAULT_NFCORPUS_ROOT) -> IrSplit:
    """Load one labelled IR split (train/dev/test).

    Cross-checks the qrels against the corpus/queries: a judgement pointing at an
    unknown doc-id or query-id is a layout error, not silently dropped — so we never
    score against dangling ids. Queries with no relevant doc are excluded (no signal).
    """
    if split not in _SPLITS:
        raise NfcorpusLayoutError(f"unknown split {split!r}; expected one of {_SPLITS}")
    corpus = _read_id_text(root / "corpus", kind="corpus")
    all_queries = _read_id_text(root / "queries", kind="queries")
    raw_qrels = _read_qrels(root / "qrels" / f"{split}.tsv")

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for qid, judged in raw_qrels.items():
        if qid not in all_queries:
            raise NfcorpusLayoutError(f"qrels {split}: query id {qid!r} absent from queries")
        relevant = {d: s for d, s in judged.items() if s >= _MIN_RELEVANT_SCORE}
        for docid in relevant:
            if docid not in corpus:
                raise NfcorpusLayoutError(f"qrels {split}: doc id {docid!r} absent from corpus")
        if not relevant:
            continue
        queries[qid] = all_queries[qid]
        qrels[qid] = judged
    if not queries:
        raise NfcorpusLayoutError(f"nfcorpus {split}: no queries with relevant docs")
    return IrSplit(split=split, queries=queries, corpus=corpus, qrels=qrels)


def load_nfcorpus_splits(
    splits: Iterable[str] = _SPLITS, root: Path = DEFAULT_NFCORPUS_ROOT
) -> dict[str, IrSplit]:
    """Load several splits at once (corpus is shared; re-read per split for simplicity)."""
    return {s: load_nfcorpus_split(s, root) for s in splits}
