# 17 — Training & Eval Log

**Single source of truth for every training run and every eval on this project.**
Whenever you train or evaluate a model, append here — this is the history + comparison
file (convention pinned in CLAUDE.md and `docs/16`). Keep entries short and comparable:
**what · how · result · good or bad · next lever**. Newest first.

Each `python -m ai.training.*` run prints a paste-ready stub at the end — drop it into
the comparison table and add a one-line judgement. Lower MAE = better; MAE/RMSE in bpm.

---

## HR encoder — comparison (PPG-DaLiA, ground-truth HR, subject-held-out)

| Date | Run | Split (train/held-out) | Encoder MAE | Classical DSP | Linear | Verdict |
|---|---|---|---|---|---|---|
| 2026-07-06 | `ppg-hr-conv-encoder@fc0e9b` | all 15 (11 / 4: S6 S8 S10 S13) | **6.50** | 11.03 | 16.86 | ✅ beats both |
| 2026-07-06 | `ppg-hr-conv-encoder@ecb9af` (demo) | 12 subj, 1500 w/subj (3 held) | 9.92 *(best 9.28)* | — | 18.66 | ⚠️ superseded |
| 2026-07-06 | `linear-hr-smoke` (Sprint 9) | S1, random-tail | — | — | 10.88 | loop proof only |

---

## Entries (newest first)

### 2026-07-06 · Sprint 10 · PPG→HR conv encoder, full-quality · `@fc0e9bceb6a2`
- **What:** 1D-CNN encoder + HR head — raw wrist BVP window → heart rate (bpm).
- **How:** all 15 subjects (11 train / 4 fully held out: S6 S8 S10 S13); 48 152 / 16 545
  windows; 200 epochs; AdamW + warmup→cosine LR, dropout 0.1 + weight-decay; **best-val
  checkpoint kept** (epoch 14). Subject-held-out (no leakage). NumPy inference (no MLX).
- **Result:** held-out HR MAE **6.50 bpm**. Classical DSP peak-detector **11.03** (100 %
  coverage, same windows); 5-stat linear **16.86**.
- **Good / bad:** ✅ beats the DoD's classical-DSP bar *and* the linear baseline
  decisively. ⚠️ overfits after ~epoch 14 (train MSE → 0.006, val flat ~6.6) — harmless
  because we keep the best checkpoint.
- **Next lever (is it best? no):** ~6.5 is good for **PPG-only** wrist HR. Biggest
  ceiling-raiser = **accelerometer fusion** (motion is what corrupts wrist PPG; likely
  → ~4–6). Then stronger regularisation / early-stop, and **PaPaGei-S pretrained-weight
  init** behind the same interface.
- Artifacts: `checkpoints/ppg-hr-conv-encoder@fc0e9bceb6a2`, `reports/…@fc0e9bceb6a2.html`.

### 2026-07-06 · Sprint 10 · encoder demo (toned-down) · `@ecb9afdb47b7` — superseded
- **What / how:** same architecture but capped (12 subjects, 1500 w/subject, 50 epochs)
  and saved the **final** epoch's weights — a fast demo, not a real run.
- **Result:** MAE 9.92 (best epoch 9.28); linear baseline 18.66.
- **Why superseded:** the caps + final-weights were quality leaks; replaced by `@fc0e9b`
  (full data, best-checkpoint). Kept for history/comparison.

### 2026-07-06 · Sprint 9 · linear HR smoke · `linear-hr-smoke`
- **What / how:** 5-stat linear head, PPG-DaLiA S1, random-tail split — harness proof only.
- **Result:** MAE 10.88 bpm. Not a shippable model; establishes the trivial bar the
  encoder must beat.

---

## Standing eval harness (classical pipeline)

Produced by `python -m scripts.run_eval` (`ai/eval_report`). Update this snapshot when an
engine changes; log real datasets when wired, gaps otherwise.

- **Deviation detection (WESAD, personal-baseline, classical HR):** F1 ≈ 0.80 / ECE ≈ 0.15
  — the bar any *learned* deviation model must beat.
- **PPG-DaLiA HR section (added Sprint 10):** classical DSP HR MAE ≈ 11 bpm on held-out
  windows; encoder MAE appears when a checkpoint is passed via `ppg_encoder_checkpoint`.
- **Retrieval / forecast / LLM-safety:** synthetic/authored smoke (see report `gaps`).
- **NFR latency/throughput:** H200-only (never measured on the Mac).

---

## How to log (do this every time)
1. **Training:** run `python -m ai.training.<trainer>`; paste its printed stub into the
   comparison table + a new dated entry; add the *good/bad* and *next-lever* judgement.
2. **Eval:** after `scripts.run_eval` or any harness change, refresh the Standing eval
   snapshot above.
3. Keep it terse — points, not prose — but never drop the key numbers, the split, or the
   verdict. Mark superseded runs rather than deleting them (history matters for comparison).
