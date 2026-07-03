# MyPHD — pre-symptomatic COVID wearables (Stanford Snyder Lab)

The illness-onset / anomaly-detection dataset for v1: continuous consumer-wearable
recordings (heart rate, steps, sleep) spanning the **baseline → pre-symptomatic →
acute-onset** window around COVID-19 infection. This is the phase-of-disease the Event
Engine's early-warning validation needs — the reason the earlier Stanford "Long COVID"
PURL (`purl.stanford.edu/cb174pb4851`) was **rejected**: that one is post-infection
(lingering-symptom) data and lacks the onset transition. See `docs/13_Datasets.md`.

- Size: ~37 GB on disk (`Phase1` ~3.4 GB, `Phase2` ~34 GB)
- License: open / research use (hosted on Google Cloud Storage by the authors)
- Provenance:
  - `Phase1/` — Mishra et al. (2020), *"Pre-symptomatic detection of COVID-19 from
    smartwatch data"* (Nat. Biomed. Eng.). Original `COVID-19-Wearables` release.
  - `Phase2/` — Alavi et al. (2021), the follow-up early-warning cohort
    (`COVID-19-Phase2-Wearables`).

**Status: downloaded and present locally** (2026-07-03), moved in from the authors'
GCS zips and extracted.

```
datasets/MyPHD/
  Phase1/   # Mishra 2020 cohort (per-participant wearable CSVs)
  Phase2/   # Alavi 2021 cohort
```

**To validate before relying on it:** confirm the per-participant file layout and the
symptom/onset label files match what the Mishra/Alavi papers describe (the ingestion
replay adapter and Event Engine eval depend on the onset labels being present) — this
has not been schema-checked yet, only placed on disk.

**Not committed** — data files are gitignored (`datasets/**`); only this README is
tracked. Supersedes the removed `datasets/stanford-longcovid/` folder.
