# 13 — Datasets

Final dataset stack for v1, decided by the team 2026-07-03. Supersedes any earlier
ad-hoc dataset notes. Two tiers, per `CLAUDE.md`'s "never invent clinical content /
never invent secrets" rule: things the agent can pull with no credential, and things
that need a human on the team (several need the doctors specifically) to obtain a
license or author content.

**Scope decision (why this list and not the TB-scale corpora):** the product runs on
the Patient State Graph + structured features + grounded retrieval + a deterministic
policy gate — not on raw foundation-model-scale training data. `BaselineEngine`'s
foundation-encoder path (PaPaGei-S/Pulse-PPG) is explicitly deferred behind the
interface, which is the strongest signal that waveform-scale training corpora
(MIMIC waveforms, UK Biobank Accel, DREAMT, MESA/SHHS full PSG) are **not** a v1
prerequisite. They're listed below as optional scale-up, not core.

## Storage layout

- **Small/open, in-repo:** `datasets/<Name>/` — data files gitignored
  (`datasets/**/*.pkl|npy|csv|zip|parquet|jsonl`), a committed `README.md` per folder
  documents provenance, license, and how to (re)fetch. Mirrors the existing
  `datasets/PPG-DaLiA/` pattern from the ingestion replay adapter.
- **Large/credentialed (>~5 GB or DUA-gated), external:** set `DATA_ROOT` (new env
  var, add to `.env` when first needed) to a path **outside** the repo and outside
  iCloud Drive (see `CLAUDE.md` operational notes — sync eviction breaks things).
  Nothing under `DATA_ROOT` is ever committed; only a path reference.
- **Server:** provisioning 512 GB persistent volume to start, expansion to 1 TB
  earmarked if a credentialed corpus (MIMIC-IV Notes, SHHS/MESA, MIMIC waveforms)
  is later pulled in for scale-up work. The core backbone below fits in ~15 GB.

## Core backbone (what v1 actually needs)

| Category | Dataset | Verified size | License | Agent-downloadable? |
|---|---|---|---|---|
| Wearables (already wired) | PPG-DaLiA | 2.7 GB | CC BY 4.0 | Yes — see below |
| Wearables | WESAD | ~1.6 GB | UCI / Research Use | Yes — Available on UCI Machine Learning Repository (zip) |
| Illness-onset labels | MyPHD Wearables (Mishra et al. / Stanford) | ~6-8 GB (zip) | Open / Research Use | Yes — Direct GCP Zip downloads |
| Sleep (deviation/baseline eval) | Sleep-EDF Expanded | 8.1 GB | ODC-Attribution 1.0 | Yes, but **deferred** — PhysioNet throttles this endpoint to ~108KB/s (~22hr); rerun `scripts/fetch_datasets.py --dataset sleep-edf` later, ideally under `caffeinate` |
| Clinical notes (gold, real EHR) | MIMIC-IV Notes | ~7 GB | PhysioNet Credentialed DUA | No — human required |
| Clinical notes (open stand-in, synthetic) | `AGBonnet/augmented-clinical-notes` (HF) | 374 MB | MIT | Yes |
| Clinical notes (open, small) | PMC-Patients (`zhengyun21/PMC-Patients`, HF) | 1.38 GB | CC BY-NC-SA 4.0 | Yes |
| Coding terminology | SNOMED CT | 370k+ concepts | Affiliate License (MLDS/NRCeS India) | No — human required |
| Coding terminology | UMLS Metathesaurus (MedCAT's base) | 3.5M+ CUIs | UTS account | No — human required |
| Coding terminology | LOINC | 109k+ codes | Free Regenstrief account | No — human required |
| Coding terminology | RxNorm | 100k+ codes | UTS account (same as UMLS) | No — human required |
| Coding terminology | ICD-10-CM | 74k+ codes | Public domain (CMS) | **Yes** |
| RAG knowledge base | `epfl-llm/guidelines` (HF) | 878 MB | mixed "other" (per-source, see dataset card) | Yes |
| RAG knowledge base | MedRAG textbooks corpus (`MedRAG/textbooks`, HF) | 212 MB | research use | Yes |
| Eval — RAG end-to-end | MIRAGE benchmark (github.com/Teddy-XiongGZ/MIRAGE) | 176 MB (full repo incl. prediction/rawdata dirs) | repo LICENSE | Yes (git clone) |
| Eval — clinical reasoning | MedQA-USMLE (`GBaker/MedQA-USMLE-4-options`, HF) | 18 MB | CC BY 4.0 | Yes |
| Eval — clinical reasoning | MedMCQA (`openlifescienceai/medmcqa`, HF) | 88 MB | Apache 2.0 | Yes |
| Eval — evidence QA | PubMedQA (`qiaojin/PubMedQA`, HF) | 300 MB | MIT | Yes |
| Eval — retrieval IR | BEIR nfcorpus (`BeIR/nfcorpus`, HF) | 3.3 MB | CC BY-SA 4.0 | Yes |
| Eval — general medical knowledge | MMLU clinical subjects (`cais/mmlu`, 5 configs) | <1 MB | MIT | Yes |

**Present locally: ~57 GB.** The HuggingFace/CMS backbone (~3.6 GB: guidelines,
MedRAG textbooks, MIRAGE, MedMCQA, PubMedQA, MedQA-USMLE, nfcorpus, MMLU-clinical
subset, ICD-10-CM, PMC-Patients, augmented-clinical-notes) plus the two manually-
downloaded wearable/onset corpora: **WESAD** (~16 GB, from UCI DOI 10.24432/C57K5T)
and **MyPHD** (~37 GB — `Phase1` Mishra 2020 + `Phase2` Alavi 2021, the pre-symptomatic
onset dataset that replaces the rejected Stanford Long-COVID PURL). All under
`datasets/`, all gitignored (only per-folder READMEs are committed). **Deferred:
Sleep-EDF** (8.1 GB, PhysioNet-throttled, see its README). Everything else in this
table needs one human action (license, account, or DUA) before it's worth fetching.

## Corrections vs. the earlier AI-generated report

Your team's research report (via Antigravity) was a good starting shortlist, but
several facts didn't hold up when checked against the live HuggingFace API,
PhysioNet, and CMS directly — recorded here so we don't re-trust the wrong number
later:

- **`epfl-llm/guidelines` is 878 MB, not "MB scale."** Still small enough to keep in
  the core backbone, just correcting the order of magnitude.
- **PubMedQA is 300 MB, not 30–50 MB** (the `pqa_artificial` split dominates).
- **"MedCorp/MedRAG" is not one thing.** It's four separate HF corpora:
  `MedRAG/pubmed` (**70 GB** — this *is* the "PubMed/PMC 100–200GB" alternative the
  team already decided to skip, confirmed correctly oversized), `MedRAG/wikipedia`
  (**45.7 GB**, not the 0.5–3 GB the report implied — if a small Wikipedia-medical
  slice is wanted later, `mvarma/medwiki` is the smaller HF dataset the report
  meant, not `MedRAG/wikipedia`; unverified, check before pulling), `MedRAG/textbooks`
  (212 MB, real, included above), `MedRAG/statpearls` (repo exists but has no
  retrievable data files via the HF API — likely served through a loader script;
  skip rather than guess).
- **WESAD:** The original origin host (`eti.uni-siegen.de`) is indeed unreliable. However, we confirmed that WESAD is officially hosted and maintained on the **UCI Machine Learning Repository** ([DOI: 10.24432/C57K5T](https://doi.org/10.24432/C57K5T)). It can be downloaded directly from there.
- **"Stanford Snyder Lab MyPHD"**: We successfully tracked down the correct pre-symptomatic (early warning) datasets from Mishra et al. (2020) and Alavi et al. (2021). They are hosted openly on Google Cloud Storage:
  - [COVID-19-Phase2-Wearables.zip](https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19-Phase2/COVID-19-Phase2-Wearables.zip)
  - [COVID-19-Wearables.zip](https://storage.googleapis.com/gbsc-gcp-project-ipop_public/COVID-19/COVID-19-Wearables.zip)
- MTSamples on Kaggle is real, but Kaggle requires an account + API token for
  scripted download (`kaggle` CLI isn't set up in this env). At 17 MB it's faster to
  download once by hand than to provision a Kaggle credential for a 17 MB file —
  treat it as a manual pull, not blocked.

## Optional scale-up (explicitly deferred, not needed to ship v1)

| Dataset | Size | Why deferred |
|---|---|---|
| MIMIC-III/IV Waveforms | 2.4 TB / TB-scale | `BaselineEngine` foundation-encoder path is deferred; classical features (v1) don't need raw waveforms at this scale |
| UK Biobank Accelerometry | 100s of TB | Population-generalization research, not a v1 product requirement |
| DREAMT | 159 GB | Bigger/redundant vs. WESAD+PPG-DaLiA for stress/affect baseline work |
| MESA (NSRR) | 385 GB | Premium PSG-grade sleep validation; Sleep-EDF (already in core backbone, open) is the defensible v1 substitute |
| SHHS (NSRR) | multi-GB | Same tradeoff as MESA — optional if Sleep-EDF proves insufficient |
| Scripps DETECT | 100s of GB | Stanford's Long COVID dataset (pending identity confirm) is the smaller illness-label substitute |
| `MedRAG/pubmed` | 70 GB | RAG KB is scoped to point-of-care guidelines + textbooks, not a general biomedical-literature index |
| `MedRAG/wikipedia` | 45.7 GB | Same reasoning; a smaller medical-Wikipedia slice can be revisited later if needed |

Pull any of these only if a specific eval or research goal needs them — log the
reason in this file when you do, so the "why" survives.

## Human-required action items (for the team, several need the doctors)

1. **PhysioNet credentialing** (MIMIC-IV Notes, and MESA/SHHS/DREAMT/MIMIC
   waveforms if the optional scale-up is ever pursued) — requires CITI "Data or
   Specimens Only Research" training, a signed DUA, and (for MIMIC) a sponsoring
   credentialed researcher. This is a per-person process; can't be automated or
   substituted.
2. **UMLS Terminology Services (UTS) account** — needed for UMLS Metathesaurus
   (MedCAT's base vocabulary) and RxNorm. Free, but requires license acceptance by
   a named individual. Once granted, hand the agent the **UTS API key** via `.env`
   (never commit it) — this unblocks the already-written but currently
   license-gated `MedCatCoder` real adapter in
   [services/doc_coding_service/coder.py](../services/doc_coding_service/coder.py).
3. **SNOMED CT Affiliate License** — via MLDS, or NRCeS for the India edition
   (relevant given the team's location). ~4–5 business day approval.
4. **LOINC** — free Regenstrief account, self-serve, no approval wait — lowest
   effort of the four terminology licenses, worth doing first.
5. **Gold-standard coded discharge summaries** — a doctor on the team needs to
   hand-code a small set (even 20–30 summaries) against SNOMED/LOINC/RxNorm to
   serve as the `DictionaryCoder`→`MedCatCoder` acceptance test set. No public
   dataset substitutes for this; it has to be your own clinicians' judgment on your
   target population.
6. **Clinical config values** — per `CLAUDE.md`, these are intentionally left as
   empty stubs and must come from clinical input, not be fabricated by the agent:
   - `config/clinical/sqi_thresholds.yaml`
   - `config/clinical/population_reference_ranges.yaml`
   - `config/clinical/event_rules.yaml`
   - `config/clinical/coding_thresholds.yaml`
7. **MyPHD Download** — The correct pre-symptomatic datasets are now documented and can be downloaded from GCP when needed.

## Reddit sentiment (spot-checked, not exhaustive)

Community sentiment on the datasets that matter most to this stack, from
r/MachineLearning, r/biostatistics, and physio/ML-adjacent discussion threads,
paraphrased:

- **WESAD** — widely used as the default wearable stress-detection benchmark in
  papers and Kaggle kernels; common complaint is the small subject count (15) limits
  generalization claims, which matches why it's paired with PPG-DaLiA rather than
  used alone.
- **MIMIC-IV** — consistently called the most realistic messy-EHR dataset available
  outside a hospital, but the credentialing/CITI process is the most-repeated
  friction point, and people flag that its ICU-heavy population skews demographics
  vs. a general outpatient population — worth remembering when validating this
  project's baseline engine against it later.
- **UMLS/SNOMED** — the recurring complaint is bureaucratic licensing overhead and
  version-churn (annual releases can shift CUIs/codes), not data quality.
- **PubMedQA / MedMCQA / MedQA** — generally accepted as reasonable proxies for
  "can this model answer medical exam-style questions," with the caveat (raised
  often) that exam-question performance doesn't strongly predict real clinical
  grounding — consistent with this project's principle that the LLM never decides,
  it only explains against the deterministic PSG.

## Fetch scripts

Auto-downloadable items are pulled by
`scripts/fetch_datasets.py` (added alongside this doc). It writes into
`datasets/<Name>/` and is safe to re-run (skips files already present).

**Gotcha for anyone writing dataset scripts:** never `import datasets` (the
HuggingFace library) while the process's working directory is the repo root — the
local `datasets/` folder shadows the pip package as a namespace package (no
`__version__`, `__file__` is `None`). Either run the script as a file with its own
directory (not `python -c` / `python -m` from repo root), or avoid the `datasets`
library entirely and use `huggingface_hub.hf_hub_download` /
`snapshot_download`, which is what `fetch_datasets.py` does.
