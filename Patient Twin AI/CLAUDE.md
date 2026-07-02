# CLAUDE.md — Patient Copilot / Twin Engine

Operating manual for the building agent. Read this every session. The full spec is in `docs/` (`00`–`12`); this file is the high-signal summary plus the rules that override convenience.

## What this is
The Patient Copilot / Twin Engine v1 — a personal-baseline "health twin" that ingests wearable + document data, maintains a per-user state graph, detects deviations from *this user's normal*, and answers grounded questions. Backend/engine only; the patient app is a thin client.

## Non-negotiable principles (never refactor away)
1. The deterministic **Patient State Engine is the primary intelligence. The LLM is explanation/orchestration only** — it proposes, never decides.
2. The **LLM never consumes raw physiological signals** — only structured features and PSG state.
3. The **Patient State Graph (PSG) is the single source of truth.** Nothing bypasses it.
4. The deterministic **Policy Engine is the last gate** before any user-facing output and can override/suppress the LLM.
5. **No closed-loop self-modification.** The LLM never edits rulesets, schemas, or routing. Those change only via human-gated versioned releases.

## Build order (follow `docs/10_Build_Backlog.md`)
Schemas/contracts → core libs (auth/consent/audit/versioning) → infra up → ingestion+SQI → baseline+PSG → events+forecast → documents+retrieval → LLM+policy+copilot → governance+eval+hardening. **Do Sprint 0 first and respect each task's Definition of Done before moving on.** Do not implement the whole spec in one pass.

## Hard rules for the agent
- **Contracts live in `schemas/` only.** Every service imports them; none redefines them. See `docs/04`.
- **Respect the stable interfaces** (`ai/interfaces/`): `BaselineEngine`, `FeatureExtractor`, `Retriever`, `Forecaster`, `LLMGateway`. Deferred components are *new implementations of these*, never new call sites.
- **Never invent clinical content.** Red-flag rules and SQI thresholds (`docs/05`, `docs/06`) and the clinical KB content are left as config stubs marked `set with clinical input`. Leave them as clearly-labelled stubs; do not fabricate thresholds, dosages, or guideline text.
- **Never invent or commit secrets.** Use `.env` (gitignored). If a credential is missing, stop and ask — do not generate one and commit it.
- **No PHI to external models.** Production LLM inference is local only. The OpenRouter path is dev-only, de-identified, and default-deny on egress (`docs/06 §6`).
- **Stamp versions** (model, ruleset, prompt, baseline-engine) on every output; emit an audit event for every mutation (`docs/04 §7`).
- Abstention and suppression are correct outcomes, not failures.

## ONE codebase, config-switched (Mac ↔ server)
There is exactly one repo. **Do not create a Mac version and a server version.** Where things run is configuration, not code.

### Inference backend (abstracted by `LLMGateway`)
| | Local dev/demo (MacBook M5 Pro 48GB) | Production (NVIDIA H200 slice) |
|---|---|---|
| LLM server | **LM Studio** (or `mlx-lm`) on the **macOS host**, port 1234 (Metal can't enter Docker) | `vLLM` (CUDA), host or container |
| Quantization | 4-bit | fp8 preferred (4-bit fallback) |
| Switch | `LLM_GATEWAY_BASE_URL` → `host.docker.internal:1234` (LM Studio) | `LLM_GATEWAY_BASE_URL` → vLLM |
The rest of the stack (Postgres, Qdrant, all services) is identical and runs in Docker on both. See `docs/08`.

### Everything non-LLM runs on the Mac unchanged
Schemas, ingestion, SQI/features, baseline engine, PSG, events, forecast, retrieval (embeddings+reranker), policy engine, API — all CPU/MPS-friendly. Build and test the full pipeline on the Mac now.

### Training backend (thin split behind one interface; shared data prep + eval)
| Target | Mac now (MLX / PyTorch-MPS) | Server later (CUDA) |
|---|---|---|
| Biosignal encoders (PaPaGei-S ~6M), task heads | ✅ train now | ✅ |
| Reranker (~0.5B), coding heads | ✅ train now | ✅ |
| **Qwen3.6 35B QLoRA** | ❌ not feasible on 48GB shared memory (and CUDA-only bitsandbytes/PEFT) | ✅ ~48–64GB, CUDA |
- Put training behind `ai/training/` with two backends (`mlx`, `cuda_qlora`) selected by `TRAIN_BACKEND`. Data prep, datasets, and eval are shared.
- **The 35B LLM is NOT fine-tuned in v1** (it runs as-is, grounded by RAG). So the Mac's inability to fine-tune it does not block v1 — defer that one job to the server.

## Environment / platform flags
```
LLM_BACKEND=mlx|vllm
LLM_GATEWAY_BASE_URL=http://host.docker.internal:8000/v1
LLM_PROFILE=local|external_deidentified|dev      # production = local
TRAIN_BACKEND=mlx|cuda_qlora
```

## Operational notes
- **Do not run this repo inside iCloud Drive / any synced folder** — sync eviction breaks git, Docker volumes, model caches, and venvs. Use a local path (e.g. `~/Developer/patient-twin-ai`).
- `.gitignore` must cover `.env`, model caches, `__pycache__/`, venvs, Docker volume data.
- Don't load-test on the Mac — it's single-tenant (~20–50 tok/s). NFR latency/throughput numbers are measured on the H200 only.
- On the Mac, confirm the MoE quant of the primary actually loads via mlx-lm before depending on it; keep `gpt-oss-20b` or a small dense model as a known-good fallback.

## Models (see `docs/03`, `docs/12`)
- Primary LLM: **Qwen3.6 35B A3B** (self-hosted; gateway-abstracted).
- Fallback/utility: **gpt-oss-20b** (weaker — utility/failover, not a quality fallback).
- Embeddings: **MedCPT + BGE**; Reranker: **bge-reranker-v2-m3**.
- OCR: **Docling + Marker**; Coding: **MedCAT** → LOINC/SNOMED/RxNorm.
- Biosignal DL: classical features in v1; **PaPaGei-S / Pulse-PPG** encoders deferred behind `FeatureExtractor`/`BaselineEngine`.
- No vision model in v1 (image analysis deferred).
```
```
