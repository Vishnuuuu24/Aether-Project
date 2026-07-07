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
| 2026-07-06 | BVP+ACC fusion (experiment) | all 15 (11 / 4, same split) | 7.77 | — | — | ❌ 19.6% worse — keep BVP-only |
| 2026-07-06 | `ppg-hr-conv-encoder@ecb9af` (demo) | 12 subj, 1500 w/subj (3 held) | 9.92 *(best 9.28)* | — | 18.66 | ⚠️ superseded |
| 2026-07-06 | `linear-hr-smoke` (Sprint 9) | S1, random-tail | — | — | 10.88 | loop proof only |

---

## Deviation — comparison (WESAD wrist BVP, stress-vs-baseline, personal-baseline)

Same signal, same windows — only the HR extractor differs (fair head-to-head). Higher
F1 / lower ECE = better. The F1 ≥ 0.80 / ECE ≤ 0.15 bar was set on *chest ECG* (clean);
wrist PPG under stress/motion is a harder signal, so read these as encoder-vs-classical.

| Date | Signal | Extractor | F1 | ECE | n | Verdict |
|---|---|---|---|---|---|---|
| 2026-07-06 | wrist BVP @64 Hz, 8 s | **encoder (learned)** | **0.701** | **0.195** | 3429 | ✅ beats classical |
| 2026-07-06 | wrist BVP @64 Hz, 8 s | classical DSP | 0.483 | 0.282 | 3425 | baseline |
| (standing) | chest ECG @700 Hz, 30 s | classical DSP | ≈0.80 | ≈0.15 | — | the DoD bar (clean signal) |

---

## Entries (newest first)

### 2026-07-06 · Sprint 10 · Accelerometer fusion — measured, **negative result**
- **What:** does adding the wrist accelerometer to raw BVP lower PPG→HR error? The
  roadmap's headline "next lever." Tested honestly rather than assumed.
- **How:** multi-channel encoder — BVP + 3-axis wrist ACC (32 Hz, linearly resampled to
  the 64 Hz BVP grid → `[512, 4]` windows). Two arms on the SAME subject-held-out split
  (held-out S6 S8 S10 S13), identical recipe, 200 epochs each. BVP-only arm = channel 0 of
  the very same windows (fair delta). `python -m ai.training.fusion_experiment`.
- **Result:** BVP-only **6.50 bpm** (reproduced the shipped model exactly) → BVP+ACC
  **7.77 bpm** — fusion **19.6 % WORSE** (Δ −1.27). Not a bug: parity-tested forward,
  identical split, same N (48 152 / 16 545).
- **Good / bad:** ✅ a clean, decisive measurement that *falsifies* the "ACC helps"
  assumption for this design — worth more than a hoped-for number. ⚠️ **do not** pursue
  naive concat-fusion. Why it fails: PPG-DaLiA GT HR is chest-ECG-derived (motion-robust),
  BVP already carries the rate, so 3 upsampled ACC channels add input noise the small CNN
  overfits (train MSE keeps dropping, val flat ~8).
- **Next lever (revised):** if ACC is ever used, it must *gate/denoise* PPG (artifact-aware
  fusion), not concat — a larger architecture, out of v1 scope. **Decision: keep BVP-only.**

### 2026-07-06 · Sprint 10 · PPG stress-context head · `ppg-stress-head@88b1e2231562`
- **What:** the DoD's *stress-context* head — a task head that reads stress vs calm from
  the SAME frozen encoder embedding that predicts HR (one embedding, two heads).
- **How:** froze `ppg-hr-conv-encoder@fc0e9b…`, embedded WESAD wrist-BVP 8 s windows
  (baseline=0 / TSST-stress=1), fit a **NumPy logistic head** (class-balanced, L2),
  **subject-held-out** (train 11 / held-out S4 S5 S13 S14). `python -m
  ai.training.train_stress_head --encoder <ckpt>`.
- **Result:** held-out **F1 0.803 · AUC 0.950 · acc 0.834** vs majority-class 0.644
  (n=912). Exposed as `stress_probability` on `FoundationEncoderFeatureExtractor`.
- **Good / bad:** ✅ the encoder embedding is strongly stress-discriminative (AUC 0.95)
  with only a linear head on top — good evidence the learned representation generalises
  beyond HR. Head is NumPy (no MLX at serving). ⚠️ WESAD stress = acute TSST; real-world
  "stress" is broader — treat as a stress-*reactivity* probe, not a deployed stress meter.
- **Next lever:** a jointly fine-tuned (not frozen-trunk) head; ACC fusion; more label
  sources than the single TSST contrast.

### 2026-07-06 · Sprint 10 · WESAD wrist-BVP deviation, encoder vs classical
- **What:** the *deviation* half of the Sprint 10 DoD — does the learned encoder beat
  the classical HR pipeline at flagging stress (TSST) vs baseline against a personal
  baseline? Scored on **WESAD wrist BVP** (PPG @ 64 Hz — the encoder's own modality;
  an earlier note wrongly called WESAD "ECG-only / different modality").
- **How:** all 15 subjects, 8 s windows (= the encoder's trained length), personal
  baseline from each subject's OWN baseline-condition windows; classical DSP and the
  learned encoder run over the **identical** windows. `python -m
  ai.training.wesad_deviation_eval --checkpoint checkpoints/ppg-hr-conv-encoder@fc0e9b…`.
- **Result:** encoder **F1 0.701 / ECE 0.195** vs classical **F1 0.483 / ECE 0.282**
  (n≈3.4k). Encoder **+0.218 F1** and better calibrated on the same signal.
- **Good / bad:** ✅ the DL path beats classical on deviation too, not just HR MAE —
  and is better calibrated. ⚠️ neither reaches the chest-ECG bar (0.80/0.15): wrist PPG
  during the TSST is genuinely noisier (motion + speech). Honest, not a regression.
- **Next lever:** accelerometer fusion (motion is the corruptor) should lift both, most
  for the encoder; then a dedicated stress head instead of routing through HR alone.

### 2026-07-06 · Sprint 10 · PPG→HR conv encoder, full-quality · `@fc0e9bceb6a2`

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

- **Deviation detection (WESAD chest ECG, personal-baseline, classical HR):** F1 ≈ 0.80 /
  ECE ≈ 0.15 — the clean-signal bar any *learned* deviation model must beat.
- **Deviation detection (WESAD wrist BVP, DL vs classical, added Sprint 10):** encoder
  F1 0.701 / ECE 0.195 vs classical 0.483 / 0.282 — the learned path wins on the same
  signal. Appears in the report as `deviation_wrist_bvp_dl` when `ppg_encoder_checkpoint`
  is configured (`python -m ai.training.wesad_deviation_eval` for the standalone run).
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
