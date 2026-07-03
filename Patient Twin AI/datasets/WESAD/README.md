# WESAD (Wearable Stress and Affect Detection)

Chest (RespiBAN) + wrist (Empatica E4) recordings, 15 subjects, stress/amusement/
neutral conditions. Used for baseline/deviation-engine validation on physiological
stress signals — see `docs/13_Datasets.md` for why this is the "best" pick for the
wearables category (paired with `datasets/PPG-DaLiA/`).

- Size: ~16 GB on disk (per-subject `.pkl` + raw signal files, 15 subjects)
- License: research use (UCI ML Repository terms)
- Source: **UCI Machine Learning Repository**, [DOI 10.24432/C57K5T](https://doi.org/10.24432/C57K5T)
  — the canonical mirror after the original `eti.uni-siegen.de` host became unreliable.
- Citation: Schmidt et al., "Introducing WESAD, a multimodal dataset for wearable
  stress and affect detection," ICMI 2018.

**Status: downloaded and present locally** (2026-07-03), extracted from the UCI zip.

```
datasets/WESAD/
  S2/  S3/  ...  S17/     # one folder per subject
```

**Not committed** — data files are gitignored (`datasets/**`); only this README is
tracked. To re-fetch on another machine, download the zip from the UCI DOI above and
extract it into this folder.
