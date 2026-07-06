# 16 — On-Device Training Roadmap (MacBook M5 Pro, 48 GB)

What we can actually *train* on the Mac now, in dependency order, without a GPU
slice and without new licensed data. Continues `15_Post_v1_Backlog.md` (which ended
at Sprint 8). This is a **plan, not an implementation** — nothing here is built
until you give the signal. Every task traces to an existing spec (`03`, `05`, `11`,
`12`, CLAUDE.md); nothing new is invented.

---

## 0. The one rule that shapes everything

Per CLAUDE.md, there is **one codebase, config-switched**. Training is no
exception: it lives behind `ai/training/` with **two backends** selected by
`TRAIN_BACKEND` (`mlx` on the Mac, `cuda_qlora` on the server). **Data prep,
dataset adapters, and evaluation are shared** across both backends — only the inner
training loop differs. We do **not** fork a "Mac trainer" and a "server trainer".

A trained model is a *new implementation of an existing stable interface*
(`FeatureExtractor`, `BaselineEngine`, `Retriever`), **never a new call site**
(CLAUDE.md; `ai/interfaces/`). The deterministic engines stay primary; a learned
model earns its way in only by beating the classical baseline on the harness we
already built (`ai/eval_report.py`, T8.2).

---

## 1. What the M5 Pro can and cannot do

Grounded in `12_Model_RAM_and_GPU_Sizing.md §3` and CLAUDE.md's training table.
48 GB is **unified** memory — shared with the OS *and* the Docker stack, so the real
training budget is ~28–34 GB after overhead, and **you cannot train while LM Studio
holds the 35B resident**.

| Target | Params | Method | Fits on M5 Pro? | Data on disk? |
|---|---|---|---|---|
| **Biosignal encoder + task heads** (PaPaGei-S / Pulse-PPG) | ~6 M | full fine-tune (PEFT unnecessary) | ✅ ~1–2 GB, comfortable | ✅ PPG-DaLiA (GT heart rate), WESAD |
| **Reranker** (bge-reranker-v2-m3) | ~0.57 B | full fine-tune (cross-encoder) | ✅ ~8–10 GB, comfortable | ✅ nfcorpus (qrels), MIRAGE, PubMedQA |
| **Embeddings** (MedCPT / BGE) — *optional* | ~0.1–0.34 B | contrastive fine-tune | ✅ ~6–10 GB | ✅ same IR corpora |
| **Document-coding heads** (MedCAT) | small | supervised | ⚠️ compute-fine, **data-BLOCKED** | ❌ SNOMED/RxNorm/LOINC/UMLS not licensed |
| **Qwen3.6 35B A3B** (instruction LoRA) | 35 B / ~3 B | QLoRA | ❌ CUDA-only (bitsandbytes/PEFT); needs ~48–64 GB dedicated | prep on Mac, **train on H200** |

**Flagship = the biosignal encoder.** It is the DL path the whole architecture was
designed to accept (`FeatureExtractor`/`BaselineEngine` DEFERRED impls), it needs no
license, and it has a concrete number to beat: the classical HR pipeline (T8.1/T8.2)
scores **F1 ≈ 0.80 / ECE ≈ 0.15** on WESAD today. That is the bar.

---

## 2. Non-negotiables (dos & don'ts for the whole track)

**Do**
- Put every trainer behind `ai/training/` with a `TRAIN_BACKEND` switch; share
  data-prep + eval (CLAUDE.md).
- Ship each trained artifact **versioned and stamped**: a new
  `baseline_engine` / `feature_extractor` / `reranker` version registered in
  `core.versioning`, stamped on every output that used it (`docs/04 §7`).
- Gate every learned model on the **existing eval harness** — it only replaces the
  classical path if it *measurably* wins; otherwise it stays behind a flag.
- Keep a **deterministic fallback**: if the encoder fails to load or a checkpoint is
  missing, the pipeline falls back to the classical extractor (fail-safe, `05 §8`).
- Train only on **on-device, research-licensed public data** (PPG-DaLiA, WESAD,
  nfcorpus). No PHI leaves the machine; no external training API (`06 §6`).
- Pin seeds, log data provenance + hyperparameters per run, and keep runs reproducible.

**Don't**
- **Don't pretrain a foundation encoder from scratch** — adapt the published
  open-weight PaPaGei-S / Pulse-PPG encoder and train task heads. Pretraining needs
  a corpus and compute we don't have.
- **Don't fine-tune the LLM on the Mac** (CUDA-only) or fine-tune anything the specs
  forbid: backbone weights, **safety policy, routing, schemas** (`03`; CLAUDE.md
  principle 5 — no closed-loop self-modification).
- **Don't invent clinical labels or thresholds** — use each dataset's own documented
  ground truth; validate layout before trusting labels (as T8.2 does).
- **Don't add a new call site** for a learned model — implement the interface.
- **Don't co-run** heavy training with the LM Studio 35B (unified-memory contention);
  free the model first.
- **Don't let a learned model bypass the Policy Engine or consent gate.**

---

## 3. Global Definition of Done (applies to every training task)

A training sprint is done only when: (1) the trainer runs under `TRAIN_BACKEND=mlx`
and produces a **versioned checkpoint**; (2) the artifact is wrapped behind its
stable interface with a **classical fallback**; (3) the **existing eval harness**
scores it and the result is recorded in `ai/eval_report.py` (real dataset label, not
synthetic); (4) it wins its stated metric bar or is honestly logged as a gap; (5)
tests + `ruff` + `mypy` pass; (6) versions are stamped and an audit event is emitted
for any artifact promotion.

---

## Sprint 9 — Training harness foundation (shared, backend-abstracted)  ✅ DONE

*Source: CLAUDE.md "Put training behind `ai/training/` with two backends"; `03 §fine-tune`.*

Build the skeleton once so every later sprint just fills in a model.

**Built:** `ai/training/` with the `TrainBackend` seam (`mlx` | `cuda_qlora`, selected
by `TRAIN_BACKEND`; MLX is a guarded import, `cuda_qlora` refuses without CUDA); a
deterministic `TrainConfig` + `set_global_seed`; a content-addressed, versioned
checkpoint writer that stamps a derived registry via `core.versioning.with_versions`
(no global mutation — human-gated release model); an eval hook re-exporting the
existing harness; and the shared `ai/eval_datasets/ppg_dalia.py` loader. The smoke
job (`python -m ai.training.smoke`) trains a trivial linear head on real PPG-DaLiA
BVP windows under MLX, emits a versioned checkpoint, and is scored (S1: HR MAE
≈ 10.9 bpm — a 5-stat linear head with no calibration; the number only proves the
loop runs, and is the trivial baseline Sprint 10's encoder must beat). `mlx>=0.18`
added in `requirements-train.txt` (Apple-silicon only). 452 tests pass; ruff clean.

- **DoD:** `ai/training/` exists with: a `TrainBackend` seam (`mlx` | `cuda_qlora`)
  selected by `TRAIN_BACKEND`; a shared `data/` layer that reuses `ai/eval_datasets`
  loaders; a `checkpoints/` writer that registers a **version** in `core.versioning`;
  a deterministic config + seed module; and an eval hook that calls the existing
  harness. A **smoke job** (train a trivial linear head on a few PPG-DaLiA windows)
  runs end-to-end under `mlx`, emits a versioned checkpoint, and is scored — proving
  the loop without depending on any real model yet.
- **Procedure:** (1) add `mlx` / `mlx-lm` + a training extra to deps (guarded import,
  so CI without MLX still imports); (2) define `TrainBackend` protocol + the two
  concrete backends (the `cuda_qlora` one may raise `NotImplementedError` on Mac);
  (3) checkpoint/version writer; (4) wire the eval harness; (5) the smoke test.
- **Do:** make MLX an optional import (skip-guard tests when absent, like the DB/Qdrant
  tests). Keep `cuda_qlora` a real class that simply refuses to run without CUDA.
- **Don't:** hard-depend on MLX at import time; don't duplicate the eval harness.

---

## Sprint 10 — Biosignal encoder + task heads  ⭐ flagship

*Source: `03 §DEFERRED`; `05 §3`; `12 §3`; `ai/interfaces/{feature_extractor,baseline_engine}.py`.*

Adapt the open-weight PaPaGei-S / Pulse-PPG encoder and train task heads on the PPG
we have ground truth for, then slot it behind the interfaces.

- **DoD:** a `FoundationEncoderFeatureExtractor` (implements `FeatureExtractor`) and a
  `FoundationEncoderBaselineEngine` (implements `BaselineEngine`) derive HR / stress-
  context from raw PPG/ECG windows via the trained encoder + heads, **behind the
  existing interfaces with a classical fallback**. On the T8.2 harness the DL path
  **matches or beats** the classical HR pipeline: HR MAE lower on PPG-DaLiA
  ground-truth HR, and deviation **F1 ≥ 0.80 / ECE ≤ 0.15** on WESAD (the current
  classical numbers). Versioned + stamped; the eval report gains a `dataset="PPG-DaLiA"`
  section and an updated WESAD one.
- **Procedure:** (1) load the pretrained encoder weights; (2) build heads — HR
  regression (PPG-DaLiA GT HR) and stress/activity context (WESAD/PPG-DaLiA labels);
  (3) train on MPS/MLX with subject-wise splits (no subject leakage across train/test);
  (4) wrap behind the two interfaces + fallback; (5) score on the T8.2 harness vs the
  classical baseline; (6) promote only if it wins.
- **Do:** fine-tune the pretrained encoder; use **subject-held-out** splits; keep the
  raw signal inside the extractor — only structured features leave it (principle 2).
- **Don't:** pretrain from scratch; leak subjects between splits; promote a model that
  doesn't beat the classical bar; change any call site.

---

## Sprint 11 — Reranker fine-tune (cross-encoder)

*Source: `03 §reranker`; `11 §component benchmarks` ("Reranker lift"); `12 §3`.*

- **DoD:** a fine-tuned `bge-reranker-v2-m3` wrapped as a new `Retriever` reranker
  implementation shows **measurable lift** (recall@k / nDCG / MRR, with vs without the
  cross-encoder) on a held-out medical IR split, via the **existing retrieval eval**
  (`ai/retrieval/eval.py`). Behind the seam; versioned; classical/lexical reranker
  stays the fallback.
- **Procedure:** (1) build training pairs from nfcorpus qrels (+ MIRAGE / PubMedQA)
  with a query→passage relevance signal; (2) full fine-tune the cross-encoder on MPS;
  (3) wrap as a `Reranker` impl; (4) A/B on the retrieval harness (with/without); (5)
  promote on lift.
- **Do:** hold out queries for eval; report the honest lift number even if small.
- **Don't:** train on the eval split; fabricate relevance labels; swap the retrieval
  call site.

---

## Sprint 12 — Document-coding heads  ⛔ BLOCKED (data, not compute)

*Source: `03 §clinical coding`; `docs/04 §4`; `15 §D` (licensed datasets).*

MedCAT coding heads (→ SNOMED CT / LOINC / RxNorm) need the **licensed terminologies
+ UMLS**, which are **not on disk** (`datasets/terminologies/` and
`datasets/MIMIC-IV-Notes/` are README-only; require DUAs). ICD-10-CM alone is present.

- **DoD (buildable now):** only the **data-prep structure** and the coding-eval
  harness skeleton, kept inert/empty until terminologies are licensed — mirroring the
  T8.3 fail-safe-when-unset pattern. **The training run itself is deferred** and logged
  as a licensed-data blocker, not attempted.
- **Don't:** fabricate code mappings or train against ICD-10-CM as if it were the full
  SNOMED/RxNorm/LOINC target.

---

## Sprint 13 — Embedding contrastive fine-tune  (optional)

*Source: `03 §embeddings`; `12 §3` (marked optional).*

Only pursue if Sprint 11's retrieval eval shows the **dense retriever** (not the
reranker) is the bottleneck. Contrastive fine-tune of MedCPT/BGE on medical pairs,
behind the embedder seam, scored on the same retrieval harness. Same DoD shape.

---

## Deferred to the H200 — Qwen3.6 35B A3B QLoRA (prep on Mac, train on server)

*Source: CLAUDE.md; `12 §3`; `15 §D`.* CUDA-only (bitsandbytes/PEFT) and ~48–64 GB —
**cannot train on the Mac**. What we *can* do now, on the Mac, is everything except
the training run:

- **DoD (Mac portion):** the `cuda_qlora` backend is wired (refuses to run without
  CUDA); a **de-identified** instruction dataset is prepared through the same shared
  data-prep; the eval harness (grounding / safety / abstention from `services/copilot_service/eval.py`)
  is ready to score adapters. The QLoRA job runs later on the H200; the Mac side is a
  clean handoff.
- **Don't:** fine-tune safety/policy/routing/schemas (only outer-loop instruction
  adapters); don't send any PHI or real patient data into the prep (`06 §6`).

---

## Recommended sequence & your decision gates

1. **Sprint 9** (foundation) — small, unblocks everything. ← *natural first build*
2. **Sprint 10** (biosignal encoder) — flagship; highest product value; clear bar to beat.
3. **Sprint 11** (reranker) — independent of 10; can run in parallel once 9 lands.
4. **Sprint 13** (embeddings) — only if 11 says so.
5. **Sprint 12** (coding) — parked on licensing.
6. **Server track** (35B QLoRA) — prep anytime; train on GPU.

**Reality checks / kill criteria** (CLAUDE.md ops notes): before depending on any
model, confirm it actually loads under MLX/MPS on this machine; if the encoder can't
beat the classical baseline after honest tuning, **keep the classical path** and log
it — abstaining from a learned model is a correct outcome, not a failure.

---

*Sprint 9 (foundation) is built. Awaiting your signal before starting Sprint 10
(the biosignal encoder) or Sprint 11 (the reranker).*
