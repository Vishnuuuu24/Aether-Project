# PPG-DaLiA (replay dataset)

Wrist (Empatica E4) + chest (RespiBAN) recordings with ground-truth heart rate,
across 15 subjects and 8 activities. Used by the dev **replay** adapter
(`services/ingestion_service/adapters/replay.py`) to stream recorded data through
ingestion as if it were live.

**Not committed** — the dataset is large and licensed for research use. Download it
yourself and drop the per-subject pickle files here:

```
datasets/PPG-DaLiA/
  S1.pkl
  S2.pkl
  ...
```

- Source: Reiss et al., "Deep PPG: Large-scale Heart Rate Estimation with
  Convolutional Neural Networks" (PPG-DaLiA), UCI ML Repository.
- Reconstructing the arrays requires `numpy` installed (`pip install numpy`); it is
  a dataset-only dependency and not needed for the unit tests.

Each subject pickle is a dict with (confirm exact keys/rates against the release):
`signal.wrist.{ACC,BVP,EDA,TEMP}`, `signal.chest.{ACC,ECG,...}`, `label`
(ground-truth HR), `activity`, `subject`. The replay adapter's sampling-rate
constants are marked *set-with-dataset* and must be verified against it.

Run once files are in place:

```
make replay DATASET=PPG-DaLiA
# or: python -m services.ingestion_service.adapters.replay --dataset PPG-DaLiA --path datasets/PPG-DaLiA/S1.pkl
```
