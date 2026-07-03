# MIRAGE (medical RAG benchmark)

End-to-end medical RAG evaluation benchmark — 7,663 questions across MMLU-Med,
MedQA-US, MedMCQA, PubMedQA, and BioASQ-Y/N. Used at release-gate time to evaluate
`ai/retrieval/eval.py` output against a standard benchmark, not just this project's
own eval queries.

- Size: 176 MB (full upstream repo, incl. `prediction/` and `rawdata/` dirs)
- Repo: https://github.com/Teddy-XiongGZ/MIRAGE — cloned into `repo/` (a nested git
  checkout; kept out of this repo's git history same as any other dataset payload)
- The `repo/benchmark.json` is the actual eval set; the underlying retrieval
  corpora it references overlap with `datasets/guidelines/` and
  `datasets/MedRAG-textbooks/` (already fetched) — no need to pull the full MedRAG
  Google-Drive corpus separately.

Fetched by `scripts/fetch_datasets.py --dataset mirage` (shallow git clone into
`datasets/MIRAGE/repo/`, kept separate from this README so the upstream repo's own
README.md never overwrites this one).
