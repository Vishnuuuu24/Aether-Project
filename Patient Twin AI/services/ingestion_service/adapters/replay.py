"""Dataset replay adapter (docs/07 §3; dev-only harness).

Streams a recorded dataset through the shared normaliser as if it were live device
data. First target: **PPG-DaLiA** (Reiss et al.) — wrist Empatica E4 + chest
RespiBAN with ground-truth HR.

The dataset itself is NOT in the repo (large; download separately — see
`datasets/PPG-DaLiA/README.md`). This module parses the released per-subject
pickle structure; the sampling rates below are marked *set-with-dataset* and must
be confirmed against that release. Unit tests exercise the stream→normalise
pipeline with a small in-memory sample of the same shape, so no download is needed
to prove the pipeline (T1.1 DoD).

Run: `python -m services.ingestion_service.adapters.replay --dataset PPG-DaLiA`
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

from schemas.reading import MeasurementContext, MetricCode

ADAPTER_NAME = "replay"
PPG_DALIA = "PPG-DaLiA"
SOURCE_DEVICE = f"dataset:{PPG_DALIA}"

# --- Sampling rates for the PPG-DaLiA release (set-with-dataset: confirm on load) ---
_HR_LABEL_FS_HZ = 0.5  # ground-truth HR: 8 s window shifted by 2 s
_WRIST_TEMP_FS_HZ = 4.0  # Empatica E4 skin temperature
_ACTIVITY_FS_HZ = 4.0

# Coarse PPG-DaLiA activity-id → measurement context. Conservative on purpose;
# refine with feature input. 0=transient 1=sitting 2=stairs 3=table-soccer
# 4=cycling 5=driving 6=lunch 7=walking 8=working.
_ACTIVITY_CONTEXT: dict[int, MeasurementContext] = {
    1: MeasurementContext.RESTING,
    5: MeasurementContext.RESTING,
    6: MeasurementContext.RESTING,
    8: MeasurementContext.RESTING,
    2: MeasurementContext.ACTIVE,
    3: MeasurementContext.ACTIVE,
    4: MeasurementContext.ACTIVE,
    7: MeasurementContext.ACTIVE,
}


def _ts(base: datetime, index: int, fs_hz: float) -> datetime:
    return base + timedelta(seconds=index / fs_hz)


def _context_at(activity: Sequence[Any], t_seconds: float) -> MeasurementContext:
    if len(activity) == 0:
        return MeasurementContext.UNKNOWN
    idx = min(int(t_seconds * _ACTIVITY_FS_HZ), len(activity) - 1)
    return _ACTIVITY_CONTEXT.get(int(activity[idx]), MeasurementContext.UNKNOWN)


def stream_ppg_dalia(
    data: Mapping[str, Any], *, patient_id: UUID, base_ts: datetime | None = None
) -> Iterator[dict[str, Any]]:
    """Yield canonical reading dicts from one PPG-DaLiA subject record.

    Emits `heart_rate` from the ground-truth label stream and `skin_temp` from the
    wrist sensor. Step/activity counts are a feature-extraction concern (T1.2), not
    fabricated here.
    """
    base = base_ts or datetime(2026, 1, 1, tzinfo=UTC)
    label: Sequence[Any] = data.get("label") or []
    activity: Sequence[Any] = data.get("activity") or []
    for i in range(len(label)):
        t_seconds = i / _HR_LABEL_FS_HZ
        yield {
            "patient_id": patient_id,
            "metric_code": MetricCode.HEART_RATE.value,
            "value": float(label[i]),
            "unit": "bpm",
            "timestamp": _ts(base, i, _HR_LABEL_FS_HZ),
            "source_device": SOURCE_DEVICE,
            "context": _context_at(activity, t_seconds).value,
            "ingest_adapter": ADAPTER_NAME,
        }

    wrist: Mapping[str, Any] = (data.get("signal") or {}).get("wrist") or {}
    temp: Sequence[Any] = wrist.get("TEMP") or []
    for i in range(len(temp)):
        yield {
            "patient_id": patient_id,
            "metric_code": MetricCode.SKIN_TEMP.value,
            "value": float(temp[i]),
            "unit": "celsius",
            "timestamp": _ts(base, i, _WRIST_TEMP_FS_HZ),
            "source_device": SOURCE_DEVICE,
            "context": MeasurementContext.UNKNOWN.value,
            "ingest_adapter": ADAPTER_NAME,
        }


def load_subject(path: str | Path) -> Mapping[str, Any]:
    """Load one PPG-DaLiA subject pickle. Requires numpy installed at runtime to
    reconstruct the arrays (a dataset-only dependency; not needed for unit tests).
    """
    with open(path, "rb") as handle:
        return cast("Mapping[str, Any]", pickle.load(handle))  # noqa: S301 — trusted local file


def stream_dataset(
    dataset: str, *, path: str | Path, patient_id: UUID, base_ts: datetime | None = None
) -> Iterator[dict[str, Any]]:
    if dataset != PPG_DALIA:
        raise ValueError(f"unknown replay dataset {dataset!r}; supported: {PPG_DALIA}")
    data = load_subject(path)
    yield from stream_ppg_dalia(data, patient_id=patient_id, base_ts=base_ts)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a recorded dataset through ingestion.")
    parser.add_argument("--dataset", default=PPG_DALIA)
    parser.add_argument(
        "--path", default=None, help="subject file; default datasets/<dataset>/S1.pkl"
    )
    parser.add_argument("--patient-id", default=None, help="target patient UUID (default: random)")
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else Path("datasets") / args.dataset / "S1.pkl"
    if not path.exists():
        print(
            f"dataset file not found: {path}\n"
            f"Download {args.dataset} and place subject files under datasets/{args.dataset}/ "
            f"(see datasets/{args.dataset}/README.md).",
            file=sys.stderr,
        )
        return 2

    patient_id = UUID(args.patient_id) if args.patient_id else uuid4()

    # Import here so the module has no import-time dependency on the normaliser path.
    from services.ingestion_service.normaliser import normalise_batch

    result = normalise_batch(
        stream_dataset(args.dataset, path=path, patient_id=patient_id),
        default_adapter=ADAPTER_NAME,
    )
    # No PHI in logs: only counts.
    print(f"normalised {len(result.accepted)} readings, {len(result.rejections)} rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
