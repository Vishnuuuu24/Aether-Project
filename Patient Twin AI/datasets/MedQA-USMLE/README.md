# MedQA-USMLE

USMLE-style multiple-choice clinical reasoning eval set. Used in `docs/11_Evaluation_Plan.md`
offline eval gates for the copilot/LLM layer — exam-question accuracy is a
sanity-check signal, not a substitute for grounding against the PSG (per
`CLAUDE.md`: the LLM never decides, it only explains).

- Size: 18 MB
- License: CC BY 4.0
- HF repo: `GBaker/MedQA-USMLE-4-options`

Fetched by `scripts/fetch_datasets.py --dataset medqa`.
