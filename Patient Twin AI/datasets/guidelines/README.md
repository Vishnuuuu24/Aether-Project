# Clinical Guidelines (RAG knowledge base)

Open-licensed clinical practice guidelines used as the primary corpus for the
hybrid-retrieval KB (`ai/retrieval/`). Point-of-care oriented, which is why the
team picked this over dumping the full PubMed/PMC OA corpus (70GB, explicitly
deferred — see `docs/13_Datasets.md`).

- Size: 878 MB (single `open_guidelines.jsonl`)
- License: mixed "other" — aggregates multiple source guideline orgs (e.g. Cancer
  Care Ontario); check the dataset card / `LICENSE` file per-source before any
  redistribution beyond internal RAG indexing.
- HF repo: `epfl-llm/guidelines`

Fetched by `scripts/fetch_datasets.py --dataset guidelines`.
