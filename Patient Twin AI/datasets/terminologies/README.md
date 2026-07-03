# Clinical coding terminologies (human-required)

All four are needed by `services/doc_coding_service` (`MedCatCoder`) and none can be
fetched by the agent — each requires a license tied to a named person on the team.
See `docs/13_Datasets.md` §"Human-required action items" for the full breakdown.

| Terminology | What to do | Effort |
|---|---|---|
| LOINC | Free Regenstrief account, self-serve download | Lowest — do first |
| UMLS Metathesaurus | UTS account + license acceptance | Medium |
| RxNorm | Same UTS account as UMLS (no separate signup) | Low once UMLS is done |
| SNOMED CT | MLDS or NRCeS (India) Affiliate License | ~4–5 business days |

Once obtained, put the **UTS API key** in `.env` (`UMLS_API_KEY`, never commit it)
— this is what unblocks the already-implemented `MedCatCoder` real adapter in
[services/doc_coding_service/coder.py](../../services/doc_coding_service/coder.py),
which currently raises `RuntimeError` until `MEDCAT_MODEL_PACK` is configured.

Raw terminology files go in **`$DATA_ROOT/terminologies/`** (external, not in the
repo) — MedCAT builds its model pack from these plus UMLS, it isn't a "drop files
here and go" folder.
