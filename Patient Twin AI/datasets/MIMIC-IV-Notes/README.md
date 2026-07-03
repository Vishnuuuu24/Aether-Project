# MIMIC-IV Notes (human-required)

Gold-standard real-world clinical notes for `doc_coding_service` validation.
**Cannot be downloaded by the agent** — requires PhysioNet credentialing.

## What the team needs to do

1. Complete CITI "Data or Specimens Only Research" training
   (https://physionet.org/about/citi-course/).
2. Register on PhysioNet, link the CITI certificate.
3. Sign the Data Use Agreement for `mimic-iv-note`.
4. Download to **`$DATA_ROOT/MIMIC-IV-Notes/`** (external, outside the repo and
   outside iCloud Drive — see `docs/13_Datasets.md`), not into this folder. Never
   commit any of it.

- Size: ~7 GB
- Source: https://physionet.org/content/mimiciv/2.2/

## Open stand-in for now

While credentialing is pending, `datasets/augmented-clinical-notes/` and
`datasets/PMC-Patients/` (both open, both already fetchable) cover unstructured
clinical-text shape well enough to build and test `doc_coding_service` end-to-end.
Swap in real MIMIC-IV notes once access is granted — same `ClinicalDocument`
schema either way.
