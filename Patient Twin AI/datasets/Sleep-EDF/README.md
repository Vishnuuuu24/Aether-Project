# Sleep-EDF Database Expanded

PSG sleep-stage recordings, open access, no credentialing. Serves as the v1 sleep
baseline/deviation-eval dataset in place of the credentialed MESA/SHHS corpora (see
`docs/13_Datasets.md` — "optional scale-up").

- Size: 8.1 GB
- License: Open Data Commons Attribution License v1.0
- Source: https://physionet.org/content/sleep-edfx/1.0.0/

Fetched by `scripts/fetch_datasets.py --dataset sleep-edf`, or manually:

```
curl -L --fail -o datasets/Sleep-EDF/sleep-edf-database-expanded-1.0.0.zip \
  https://physionet.org/content/sleep-edfx/get-zip/1.0.0/
unzip -q datasets/Sleep-EDF/sleep-edf-database-expanded-1.0.0.zip -d datasets/Sleep-EDF/
```

**Status: not fetched.** PhysioNet server-side throttles this specific endpoint to
~108 KB/s (confirmed — the same machine gets ~19 MB/s to other hosts), which makes
the 8.1 GB file a ~22 hour download. Deferred rather than run overnight since
nothing in the current sprint blocks on it. Re-run the command above (or
`--dataset sleep-edf`) whenever it's next needed; consider `caffeinate` if kicking
it off unattended so macOS sleep doesn't stall it.

**Not committed** — data files are gitignored (`datasets/**`), only this README is
tracked.
