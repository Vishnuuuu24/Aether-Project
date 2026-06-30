# 03 — Tech Stack & Model Registry

All model identifiers below were verified against the live OpenRouter catalogue. Self-hosting uses the HuggingFace repo; OpenRouter slugs are the dev-fallback route through the LLM Gateway.

## 1. Runtime stack (pinned)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | |
| API | FastAPI + Uvicorn | async; OpenAPI auto-gen |
| ML | PyTorch, Transformers, PEFT | PEFT for later LoRA only |
| LLM serving | **vLLM** (host GPU) | OpenAI-compatible endpoint |
| Relational DB | PostgreSQL 16 | FHIR resources + OMOP mirror + PSG tables |
| Vector DB | Qdrant | dense vectors for hybrid RAG |
| Sparse search | BM25 | Postgres `tsvector` (v1) or OpenSearch (later) |
| Queue/stream | Redis Streams (v1) | ingestion buffering, backpressure |
| Object store | MinIO (S3-compatible) | documents, OCR artefacts, model cache |
| Auth | OAuth2 / JWT, RBAC | pseudonymous UUID7 subjects |
| Containerisation | Docker + Compose | see `08` |
| Observability | OpenTelemetry + Prometheus + Grafana | traces include model+ruleset versions |

## 2. Model registry

### 2.1 Primary LLM
- **Qwen3.6 35B A3B** — sparse MoE, 35B total / ~3B active, multimodal-capable, 262k context.
  - HF: `Qwen/Qwen3.6-35B-A3B`
  - OpenRouter (dev fallback): `qwen/qwen3.6-35b-a3b`
  - Serving: vLLM, **4-bit AWQ** `‹GPU-DEP›`, `MAX_MODEL_LEN` default 32768, single model resident.
  - Role: explanation + orchestration only. RAG-grounded. Proposes; never decides.
  - Note: the A3B sparsity buys **latency/throughput, not memory** — all experts stay resident, so budget for the full ~35B at the chosen precision.

### 2.2 Utility / dev fallback
- **gpt-oss-20b** — `openai/gpt-oss-20b` (HF `openai/gpt-oss-20b`), Apache-2.0, text-only, 21B/3.6B MoE.
  - Role: cheap utility tasks + failover during dev. **Caveat:** materially weaker than the primary (low intelligence index) — do not let clinical-explanation quality silently degrade to it in production. Production failover should route to a stronger hosted model via the Gateway (de-identified) or queue/abstain.

### 2.3 Vision
- **None in v1.** Image/scan analysis is deferred. Qwen3.6 is natively multimodal, so if a single image task appears later it can be handled by the primary without adding a resident vision model. A dedicated medical-imaging model (MedGemma-class) is `DEFERRED`.

### 2.4 Embeddings (hybrid RAG)
- **MedCPT** (medical bi-encoder) — primary dense retriever for clinical text.
- **BGE** (general embeddings) — general-text fallback / second view.
- Both indexed in Qdrant; choice per-collection.

### 2.5 Reranker
- Cross-encoder medical reranker (MedCPT cross-encoder or BGE-reranker class). Runs on top-k from hybrid retrieval.

### 2.6 Document & coding
- **OCR / layout:** Docling + Marker (text documents only in v1).
- **Clinical coding:** MedCAT → SNOMED CT, LOINC, RxNorm.

### 2.7 Biosignal DL stack
- v1 runtime: **classical SQI + physiological feature extraction** only.
- `DEFERRED` behind `FeatureExtractor`/`BaselineEngine`: biosignal foundation encoder (**PaPaGei-S**, **Pulse-PPG** as the practical open-weight options), shared physiological embeddings, task heads. Not loaded in v1.

## 3. Fine-tuning policy (from registry decisions)

- **Do NOT fine-tune in v1:** backbone weights, safety policy, routing, schemas.
- **Fine-tune later (outer loop only):** instruction adapters (LoRA on the primary), the reranker, document-coding heads, physiological task heads.
- QLoRA on a 4-bit MoE is finicky (router/adapter interaction under quantization) — when this is attempted later, gate it behind an explicit eval (`11`).

## 4. VRAM budget `‹GPU-DEP›`

Targeting the stated budget (ideal 40–50 GB; max 100 GB). Approximate footprints:

| Component | 4-bit footprint | Notes |
|---|---|---|
| Qwen3.6 35B A3B weights | ~18–22 GB | all experts resident |
| KV cache @ 32k ctx, low concurrency | ~4–10 GB | scales with ctx × concurrency |
| MedCPT + BGE embedders | < 2 GB | can run CPU if needed |
| Cross-encoder reranker | < 2 GB | |
| **Total (no vision)** | **~28–36 GB** | fits the 40–50 GB ideal with headroom |

## 4a. Serving platforms (confirmed)

Two target platforms, treated as two environments behind the **same** OpenAI-compatible LLM Gateway. Switching between them is a config change (`LLM_GATEWAY_BASE_URL`), not a code change.

### Production — NVIDIA H200 DGX slice (80–100 GB) — CUDA / vLLM
- Serving engine: **vLLM** (CUDA). Quantization: **fp8** preferred at this budget (better quality than 4-bit; H200 has native fp8); 4-bit AWQ is the fallback.
- `MAX_MODEL_LEN`: 32k → 128k feasible; real concurrency via PagedAttention.
- Containerised GPU works (NVIDIA Container Toolkit) — the in-Compose `vllm` profile in `08 §4` applies.
- Embeddings + reranker can be co-resident on-GPU comfortably.
- **This is the production serving target.** PHI-allowed (inside the trust boundary).

### Local dev / demo — MacBook M5 Pro 48 GB (unified memory) — Metal / mlx-lm
- Serving engine: **mlx-lm** running **natively on the macOS host** (not in Docker — Metal GPU cannot be passed into containers). `vllm-metal` exists but its MoE expert-routing support is unconfirmed for models like this; prefer `mlx-lm` and verify the `mlx-community` 4-bit quant of Qwen3.6 35B A3B loads first. Keep `gpt-oss-20b` or a small dense model as a known-good fallback.
- Unified memory is shared with the OS **and** the Docker stack. Budget after overhead: ~28–34 GB for the model. 4-bit A3B (~18–20 GB) fits; keep `MAX_MODEL_LEN` modest (≤16–32k), embeddings/reranker small or on CPU, and don't co-run heavy services.
- Expect ~20–50 tok/s, single-tenant. Fine for building and demos; **not** a multi-user server.
- Two valid modes: (a) **private local** — mlx-lm on host, PHI stays on the machine (the "sovereign" mode); (b) **dev** — route the gateway to OpenRouter, but only with synthetic/de-identified data (the de-id egress gate in `06 §6` still applies).

### Why no card-agnostic CUDA advice
The earlier "40–48 GB RTX/A100" framing is dropped — neither confirmed option is a generic discrete CUDA card. The footprint table above still governs the H200 path. On the Mac, memory is unified and shared, so the table is a floor, not a budget.

## 5. Versioning

Every deployed artefact (model weights+quant, prompt template, ruleset, schema) carries a semantic version recorded in the Governance version registry. Every output records the exact versions that produced it (NFR-6).
