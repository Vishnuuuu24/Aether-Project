# PubMedQA

Evidence-based yes/no/maybe QA over PubMed abstracts. Used to eval the retrieval +
answer-grounding pipeline's ability to stick to evidence rather than hallucinate.

- Size: 300 MB (dominated by the `pqa_artificial` split; `pqa_labeled` alone is
  ~1 MB if only the human-labeled gold split is needed)
- License: MIT
- HF repo: `qiaojin/PubMedQA`

Fetched by `scripts/fetch_datasets.py --dataset pubmedqa`.
