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
| 2026-07-07 | `papagei-s-hr-encoder@5192b7` | all 15 (11 / 4: S6 S8 S10 S13) | **4.59** | 10.63 | 16.85 | ✅ **best** — real PaPaGei-S pretrained init, −29% vs from-scratch |
| 2026-07-06 | `ppg-hr-conv-encoder@fc0e9b` | all 15 (11 / 4: S6 S8 S10 S13) | 6.50 | 11.03 | 16.86 | ✅ beats both (from-scratch) |
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
| 2026-07-07 | wrist BVP →125 Hz, 10 s | **PaPaGei-S (finetuned)** | **0.701** | **0.195** | 2737 | ✅ beats classical (+0.227); **= from-scratch; still < 0.80** |
| 2026-07-07 | wrist BVP →125 Hz, 10 s | classical DSP | 0.474 | 0.278 | 2736 | baseline @125 Hz |
| 2026-07-06 | wrist BVP @64 Hz, 8 s | encoder (from-scratch) | 0.701 | 0.195 | 3429 | ✅ beats classical |
| 2026-07-06 | wrist BVP @64 Hz, 8 s | classical DSP | 0.483 | 0.282 | 3425 | baseline @64 Hz |
| (standing) | chest ECG @700 Hz, 30 s | classical DSP | ≈0.80 | ≈0.15 | — | the DoD bar (clean signal) |

---

## Entries (newest first)

### 2026-07-07 · Sprint 11 · Reranker fine-tune — **BUILT + gate-green; RUN deferred → H200** ⏳
- **What:** the full Sprint 11 reranker pipeline (`bge-reranker-v2-m3` fine-tune on BEIR
  nfcorpus medical IR) — loader, qrels fetch, BM25-hard-negative pair builder, A/B lift
  eval (reuses `ai/retrieval/eval.py` metrics), `FineTunedReranker` seam impl (versioned,
  lexical fallback), and the full-quality trainer. All code + tests landed, ruff/mypy
  clean, fast suite green. **No fine-tuned checkpoint yet — the training RUN is deferred.**
- **How / why deferred:** measured real MPS throughput on the 568M cross-encoder =
  **~5.0 s/step** at batch 16 / seq 512 (isolated fwd+bwd probe; the first attempt's
  "257 s/step" was laptop-sleep time miscounted by tqdm). Mixed precision made it *worse*
  on MPS (fp16 6.4 s, bf16 7.8 s vs fp32 4.9 s) — no speedup lever. Uncapped nfcorpus is
  ~48 h/**epoch**; even the quality-capped recipe is ~18 h on the Mac. Decision (user):
  run it on the **H200** where the same script (CUDA auto-selected) finishes in minutes.
- **Recipe locked for the run:** all 2590 train queries · positives capped 10/query
  (drops the noisy broad-query tail — a handful of queries mark hundreds–1363 of 3633
  docs relevant; quality control, not a data cap) · 4 BM25 hard negatives · 2 epochs ·
  best-val on a 200-query DEV subset · honest lift on held-out **TEST** with the full
  relevant set. `python -m ai.training.train_reranker` (`--smoke` first to validate the
  tail in seconds on CUDA).
- **Good / bad:** ✅ clean Mac↔server split — the flagship-slow job moves to the GPU that
  exists for it, with zero code change and full plumbing pre-validated. ⏳ result pending
  GPU access; DoD (measurable held-out lift) will be filled here post-run.
- **Next lever:** on the box — `--smoke`, then the full run; paste the printed stub here.

### 2026-07-07 · Sprint 10 · **PaPaGei-S pretrained encoder, fine-tuned** · `papagei-s-hr-encoder@5192b7651ac8`
- **What:** the DoD's literal "load the pretrained encoder weights / fine-tune the
  pretrained encoder" — closes the gap a review flagged (we had trained from scratch).
  The AUTHENTIC PaPaGei-S (Nokia Bell Labs, ICLR'25; Zenodo 10.5281/zenodo.13983110,
  MD5-verified, BSD-3-Clause-Clear) — an 18-block 1-D ResNet, 512-d embedding.
- **How:** ported the trunk to NumPy for serving (parity vs the real torch model =
  **5.8e-15**, machine precision, float64) and FULL-fine-tuned it in PyTorch/MPS
  (CLAUDE.md-sanctioned Mac backend for PaPaGei-S) on PPG-DaLiA BVP resampled to
  PaPaGei's native 125 Hz / 10 s contract. Fresh HR head (head LR 10×), subject-held-out
  (S6 S8 S10 S13 — the SAME split as the from-scratch run), best-val checkpoint kept,
  60 epochs, all 15 subjects. Serving is NumPy (a post-fine-tune parity check gates the
  export). `python -m ai.training.train_papagei_encoder`.
- **Result:** held-out **HR MAE 4.59 bpm** (best epoch 56, RMSE 9.53) vs from-scratch
  **6.50**, classical DSP **10.63**, linear **16.85** — a **−29% error reduction over the
  from-scratch encoder**. Promotion advisory: RECOMMENDED (beats both bars); human-gated.
- **Good / bad:** ✅ the pretrained init is a decisive, honest win on HR — exactly where
  PaPaGei's pretraining objective lives. The NumPy serving port matching the torch model
  to machine precision means the shipped model IS the validated model. ⚠️ this does NOT
  by itself clear the deviation bar (see next entry) — better HR ≠ better stress
  separability.
- **Next lever:** the deviation bar is signal-limited, not encoder-limited (below).

### 2026-07-07 · Sprint 10 · PaPaGei-S deviation + stress-head (honest bar check)
- **What:** does the pretrained encoder push WESAD wrist-BVP deviation past the literal
  DoD bar (F1 ≥ 0.80 / ECE ≤ 0.15)? The review's hypothesis was "F1 0.701 is low because
  it lacks the pretrained encoder." Tested directly.
- **How:** scored the fine-tuned PaPaGei extractor vs classical DSP on the SAME resampled
  125 Hz / 10 s wrist-BVP windows (`wesad_deviation_eval --papagei-checkpoint …`); trained
  a stress head on the fine-tuned embedding, subject-held-out (`train_stress_head
  --papagei`).
- **Result:** deviation **F1 0.701 · ECE 0.195** (n=2737) — beats classical@125 Hz
  (0.474) by **+0.227**, but lands at the **SAME F1 as the from-scratch encoder** and
  **stays below 0.80**. Stress head **F1 0.775 · AUC 0.917 · acc 0.818** (majority 0.643,
  n=729) — slightly *behind* the from-scratch stress head (0.803 / 0.950).
- **Good / bad:** ✅ this DISPROVES the review's causal hypothesis: even the real
  pretrained SOTA encoder — which cut HR MAE by 29% — leaves deviation F1 unchanged at
  0.701. The 0.80 gap is therefore **signal-limited, not encoder-limited**: wrist PPG
  under the TSST (motion + speech) is a genuinely harder signal than the clean chest ECG
  the 0.80 bar was set on. Two independent encoders (from-scratch + PaPaGei) converge to
  0.701. ⚠️ PaPaGei is not uniformly better — on stress *classification* it's marginally
  behind the from-scratch embedding, so it is NOT a strict replacement; it's the HR
  encoder of choice, comparable on stress.
- **Next lever:** to move the deviation bar you'd need a cleaner signal (chest ECG stays
  classical at ≈0.80) or a deviation model trained end-to-end on the stress label — not a
  better HR encoder. Promotion is human-gated regardless.

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
