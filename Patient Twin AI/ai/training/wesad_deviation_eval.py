"""WESAD wrist-BVP deviation eval — classical vs learned encoder (docs/16 Sprint 10).

    python -m ai.training.wesad_deviation_eval
    python -m ai.training.wesad_deviation_eval --checkpoint checkpoints/ppg-hr-conv-encoder@<hash>

Scores the personal-baseline deviation task (baseline-condition vs stress/TSST) on
WESAD **wrist BVP @ 64 Hz** — the SAME modality the Sprint 10 encoder was trained on.
Both the classical DSP extractor and the learned `FoundationEncoderFeatureExtractor`
run over identical 8 s windows, so the F1 / ECE difference is attributable to the
encoder alone. This is the honest DL half of the deviation DoD (the classical bar is
F1 ≥ 0.80 / ECE ≤ 0.15, measured on chest ECG — wrist PPG is harder under motion).

Promotion is NOT automatic (CLAUDE.md principle 5): this prints scores + a docs/17
stub; a human runs the gate and decides.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from ai.baseline.eval import (
    DetectionMetrics,
    detection_metrics,
    expected_calibration_error,
)
from ai.eval_datasets.wesad import (
    WesadLayoutError,
    load_wesad_wrist_bvp_labelled_deviations,
    wesad_available,
)
from ai.features.foundation_encoder import FoundationEncoderFeatureExtractor
from ai.features.waveform_extractor import WaveformFeatureExtractor
from ai.interfaces.feature_extractor import FeatureExtractor

DEFAULT_WESAD_ROOT = Path("datasets/WESAD")
# Chest-ECG classical bars from docs/17 (the numbers any learned path is measured against).
CLASSICAL_F1_BAR = 0.80
CLASSICAL_ECE_BAR = 0.15


@dataclass(frozen=True)
class ArmResult:
    """One extractor's scores on the wrist-BVP deviation task."""

    label: str
    detection: DetectionMetrics
    ece: float

    @property
    def f1(self) -> float:
        return self.detection.f1


def _score_arm(
    label: str,
    root: Path,
    extractor: FeatureExtractor,
    *,
    subjects: list[str] | None,
    window_seconds: float,
    max_subjects: int | None,
    target_fs_hz: float | None = None,
) -> ArmResult:
    labelled = load_wesad_wrist_bvp_labelled_deviations(
        root,
        extractor=extractor,
        subjects=subjects,
        window_seconds=window_seconds,
        max_subjects=max_subjects,
        target_fs_hz=target_fs_hz,
    )
    return ArmResult(
        label=label,
        detection=detection_metrics(labelled),
        ece=expected_calibration_error(labelled).ece,
    )


def _print_arm(arm: ArmResult) -> None:
    d = arm.detection
    print(
        f"  {arm.label:<20} F1 {d.f1:.3f}  "
        f"(P {d.precision:.3f} / R {d.recall:.3f})  ECE {arm.ece:.3f}  "
        f"[tp {d.tp} fp {d.fp} fn {d.fn} tn {d.tn}, n={d.n}]"
    )


def _score_papagei_block(
    root: Path,
    papagei_checkpoint: Path,
    *,
    subjects: list[str] | None,
    max_subjects: int | None,
) -> None:
    """Score the fine-tuned PaPaGei-S extractor vs classical DSP on the SAME resampled
    125 Hz / 10 s wrist-BVP windows (the PaPaGei pretrained-input contract). This is the
    honest test against the literal DoD bar (F1 ≥ 0.80 / ECE ≤ 0.15).

    Deviation scoring reads only `heart_rate_bpm`, so a stress head is deliberately NOT
    loaded here — it would compute `stress_probability` that this eval never consumes,
    which would falsely imply it affects F1/ECE. The stress head is evaluated in its own
    entry point (`train_stress_head.py`)."""
    from ai.features.papagei_extractor import PapageiFeatureExtractor

    print("\n── PaPaGei-S arm (resampled 125 Hz / 10 s windows) ──")
    classical = _score_arm(
        "classical DSP (125Hz)", root, WaveformFeatureExtractor(),
        subjects=subjects, window_seconds=10.0, max_subjects=max_subjects,
        target_fs_hz=125.0,
    )
    _print_arm(classical)
    extractor = PapageiFeatureExtractor.from_checkpoint(papagei_checkpoint)
    papagei = _score_arm(
        "PaPaGei-S (finetuned)", root, extractor,
        subjects=subjects, window_seconds=10.0, max_subjects=max_subjects,
        target_fs_hz=125.0,
    )
    _print_arm(papagei)
    delta = papagei.f1 - classical.f1
    verb = "beats" if delta > 0 else ("ties" if abs(delta) < 1e-9 else "trails")
    print(f"PaPaGei-S {verb} classical on wrist-BVP F1 by {delta:+.3f}")
    meets = papagei.f1 >= CLASSICAL_F1_BAR and papagei.ece <= CLASSICAL_ECE_BAR
    print(
        f"vs literal DoD bar (F1 ≥ {CLASSICAL_F1_BAR:.2f} / ECE ≤ {CLASSICAL_ECE_BAR:.2f}): "
        f"{'✅ MEETS' if meets else '❌ below'} "
        f"(F1 {papagei.f1:.3f}, ECE {papagei.ece:.3f})"
    )
    print("\n── log entry (paste into docs/17_Training_Log.md) ──")
    print(f"| PaPaGei-S deviation | F1 {papagei.f1:.3f} · ECE {papagei.ece:.3f} "
          f"(classical@125Hz F1 {classical.f1:.3f}) |")


def run(
    root: Path,
    *,
    checkpoint: Path | None,
    papagei_checkpoint: Path | None = None,
    subjects: list[str] | None = None,
    window_seconds: float = 8.0,
    max_subjects: int | None = None,
) -> int:
    if not wesad_available(root):
        print(f"WESAD not found under {root} — nothing to score.", file=sys.stderr)
        return 2

    print(f"WESAD wrist-BVP deviation eval (window {window_seconds:.0f}s, PPG @ 64 Hz)")
    classical = _score_arm(
        "classical DSP",
        root,
        WaveformFeatureExtractor(),
        subjects=subjects,
        window_seconds=window_seconds,
        max_subjects=max_subjects,
    )
    _print_arm(classical)

    encoder: ArmResult | None = None
    if checkpoint is not None:
        extractor = FoundationEncoderFeatureExtractor.from_checkpoint(checkpoint)
        encoder = _score_arm(
            "encoder (learned)",
            root,
            extractor,
            subjects=subjects,
            window_seconds=window_seconds,
            max_subjects=max_subjects,
        )
        _print_arm(encoder)
        delta = encoder.f1 - classical.f1
        verb = "beats" if delta > 0 else ("ties" if abs(delta) < 1e-9 else "trails")
        print(f"\nencoder {verb} classical on wrist-BVP F1 by {delta:+.3f}")
    else:
        print("\n(no --checkpoint: classical arm only; pass a checkpoint to score the encoder)")

    print(
        f"\nreference bars (chest ECG, docs/17): F1 ≥ {CLASSICAL_F1_BAR:.2f} / "
        f"ECE ≤ {CLASSICAL_ECE_BAR:.2f}. Wrist PPG under motion is a harder signal — "
        "report the honest number, don't force the bar."
    )
    _print_log_stub(classical=classical, encoder=encoder, window_seconds=window_seconds)

    if papagei_checkpoint is not None:
        _score_papagei_block(
            root, papagei_checkpoint, subjects=subjects, max_subjects=max_subjects,
        )
    return 0


def _print_log_stub(
    *, classical: ArmResult, encoder: ArmResult | None, window_seconds: float
) -> None:
    print("\n── log entry (paste into docs/17_Training_Log.md, then add judgement) ──")
    print(f"| task | WESAD wrist-BVP deviation (baseline vs stress), {window_seconds:.0f}s win |")
    print(f"| classical DSP | F1 {classical.f1:.3f} · ECE {classical.ece:.3f} |")
    if encoder is not None:
        print(f"| encoder (learned) | F1 {encoder.f1:.3f} · ECE {encoder.ece:.3f} |")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="WESAD wrist-BVP deviation eval.")
    parser.add_argument("--root", default=str(DEFAULT_WESAD_ROOT))
    parser.add_argument("--checkpoint", default=None, help="from-scratch encoder ckpt (optional)")
    parser.add_argument("--papagei-checkpoint", default=None,
                        help="fine-tuned PaPaGei-S ckpt (scores the 125Hz/10s arm vs the DoD bar)")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        return run(
            Path(args.root),
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            papagei_checkpoint=(
                Path(args.papagei_checkpoint) if args.papagei_checkpoint else None
            ),
            window_seconds=args.window_seconds,
            max_subjects=args.max_subjects,
        )
    except WesadLayoutError as exc:
        print(f"WESAD layout error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
