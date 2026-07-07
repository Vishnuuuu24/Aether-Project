# BEIR nfcorpus

Small consumer-health information-retrieval benchmark (BEIR suite). Used to eval
`ai/retrieval/` (BM25 + dense + rerank fusion) against a standard IR benchmark
rather than only this project's own eval queries, and as the labelled medical-IR
signal for the **Sprint 11 cross-encoder reranker** fine-tune + lift eval
(`ai/eval_datasets/nfcorpus.py`, `ai/training/train_reranker.py`).

- Size: 3.3 MB corpus/queries + ~1 MB qrels
- License: CC BY-SA 4.0
- HF repo: `BeIR/nfcorpus` (corpus + queries); `BeIR/nfcorpus-qrels` (relevance)

Layout:
- `corpus/…parquet`, `queries/…parquet` — `_id`, `title`, `text`.
- `qrels/{train,dev,test}.tsv` — human relevance judgements (`query-id`,
  `corpus-id`, graded `score` 1–2). BeIR/nfcorpus ships only corpus+queries; the
  qrels come from the sibling `BeIR/nfcorpus-qrels` repo.

Fetched by `scripts/fetch_datasets.py --dataset nfcorpus` (which pulls the qrels
sibling repo too).
