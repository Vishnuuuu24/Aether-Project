# ICD-10-CM

Public-domain CMS code set. Not a primary coding target for `doc_coding_service`
(SNOMED/LOINC/RxNorm are, per `CLAUDE.md`/`docs/06`) but useful for
crosswalk/back-compat mapping and is trivially open, so it's included in the core
backbone at near-zero cost.

- Size: 2.2 MB (zip)
- License: Public domain
- Source: https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip
  (URL includes the year; re-check on cms.gov if this 404s in a future year)

Fetched by `scripts/fetch_datasets.py --dataset icd10cm`.
