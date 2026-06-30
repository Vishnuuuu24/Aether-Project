# 12 — Model RAM & GPU Sizing (NVIDIA request)

**Purpose:** exact models, parameter counts, and VRAM estimates for (a) single-user testing inference, (b) the full product running together after fine-tuning, and (c) fine-tuning. Figures are **generous (upper-bound) estimates including framework + KV + overhead**, not theoretical floors. Use these to size the GPU request.

## Assumptions
- **Single-user / test mode** (one concurrent request, LM Studio-style load-and-run), not production concurrency.
- VRAM figures include weights + KV cache (32k ctx for LLMs) + CUDA/framework overhead.
- "Test precision" = the precision you'd actually load for testing; `fp8`/`fp16` columns show quality-headroom cost.
- Fine-tuning jobs run **one at a time**, so the fine-tuning envelope is the single largest job, not their sum.

## 1. Inference — per model, single-user

| Model | Role | Params (total / active) | Test precision | VRAM (test) | fp8 | fp16 |
|---|---|---|---|---|---|---|
| **Qwen3.6 35B A3B** | Primary LLM | 35B / ~3B (MoE) | 4-bit | **~28 GB** | ~40 GB | ~78 GB |
| **gpt-oss-20b** | Fallback/utility LLM | 21B / ~3.6B (MoE) | MXFP4 (~4-bit) | **~16 GB** | ~24 GB | ~44 GB |
| **MedCPT + BGE** | Embeddings | ~0.11B + ~0.34B | fp16 | **~2 GB** | — | — |
| **bge-reranker-v2-m3** | Cross-encoder reranker | ~0.57B | fp16 | **~1.5 GB** | — | — |
| **Docling + Marker/Surya** | OCR (on-demand) | mixed (~100s M) | fp16 | **~3.5 GB** | — | — |
| **MedCAT** | Clinical coding | small | CPU / fp16 | **~1 GB** (CPU-able) | — | — |
| **PaPaGei-S / Pulse-PPG + heads** | Biosignal DL encoder | ~6M | fp16 | **<0.5 GB** | — | — |

MoE note: only ~3B of 35B activate per token, which buys **speed, not memory** — all experts stay resident, so size for the full parameter count.

## 2. Inference — full product running together (single-user, after fine-tuning)

Fine-tuned LoRA adapters add <0.5 GB on top of the base (or zero if merged).

| Configuration | VRAM | Note |
|---|---|---|
| Primary LLM (4-bit) + full retrieval/doc/biosignal stack | **~42–44 GB** | fallback LLM not co-resident (demand-loaded/routed) |
| Same, with **gpt-oss-20b co-resident** (worst case) | **~58–60 GB** | only if you keep both LLMs hot at once |
| **Recommended operating point:** primary at **fp8** (quality) + full stack | **~48–50 GB** | production-quality single-user; fallback demand-loaded |

## 3. Fine-tuning plan & VRAM (QLoRA / efficient)

**Will fine-tune** (outer loop, never the safety/policy/routing/schemas):

| Target | Method | Est. peak VRAM (optimal seq/batch) |
|---|---|---|
| **Qwen3.6 35B A3B** (instruction adapters) | **QLoRA** — 4-bit frozen base + bf16 adapters + paged AdamW + grad checkpointing | **~48 GB** (up to ~64 GB for long seq / larger batch) |
| **Reranker** (bge-reranker) | Full fine-tune (small) | ~8–10 GB |
| **Document coding** (MedCAT / coding head) | Supervised | ~2–4 GB |
| **Biosignal encoder + task heads** (PaPaGei-S/Pulse-PPG) | Full fine-tune (PEFT unnecessary at ~6M params) | ~1–2 GB |
| Embeddings (MedCPT/BGE) — *optional* | Contrastive fine-tune | ~6–10 GB |

**Will NOT fine-tune in v1:** gpt-oss-20b (utility/fallback; QLoRA ~24–28 GB only if ever needed), and the primary's backbone weights, safety policy, routing, and schemas.

**Fine-tuning envelope = the single largest job = Qwen3.6 QLoRA ≈ 48 GB (64 GB generous).**

## 4. GPU recommendation (the ask)

Binding constraints: full-stack inference with fallback co-resident (~60 GB), 35B QLoRA (~48–64 GB), and fp8 quality headroom (~50 GB). Adding ~25–30% operating headroom:

| Tier | VRAM | What it gets you |
|---|---|---|
| Minimum viable | **64 GB** | full stack OR a 35B QLoRA job, one at a time, 4-bit |
| **Recommended** | **80 GB** (H100 / H200 80 GB slice) | full stack **+** co-resident fallback **+** 35B QLoRA **+** fp8 primary, comfortably |
| Optimal | **96–141 GB** (H200 slice / full H200) | fp8 primary at long context, co-resident fallback, QLoRA with generous seq/batch **simultaneously**, and room for the deferred vision/medical model |

**Request: an H200 slice of ≥ 80 GB, ideally 96 GB+.** Justification: (1) the full single-user product fits in ~44–60 GB, (2) fine-tuning the 35B needs ~48–64 GB, (3) fp8 gives production-grade quality with headroom, (4) ≥96 GB future-proofs for concurrency, longer context, and the medical-imaging model that is deferred from v1.

## 5. Caveats
- Biosignal model sizes from published papers (PaPaGei-S 5.7M; Pulse-PPG/AnyPPG encoders ~5.85M) — negligible either way.
- LLM KV cache scales with context × concurrency; figures assume 32k single-user. Production concurrency is sized separately (out of scope here).
- gpt-oss-20b is materially weaker than the primary; treat it as utility/failover, not a quality fallback.
