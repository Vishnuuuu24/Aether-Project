# 08 — Docker Stack

Goal: everything stateless + Postgres + Qdrant containerised; the model runs on the host GPU and the Dockerised services route to it. An optional Compose **profile** runs vLLM in a container for single-host GPU machines.

## 1. Topology

```mermaid
flowchart LR
  subgraph host["Host machine (GPU)"]
    VLLM[vLLM server :8000<br/>Qwen3.6 35B A3B 4-bit]
  end
  subgraph net["Docker network: hasa-net"]
    GWAPI[api-gateway :8080]
    PSE[patient-state-engine]
    ING[ingestion-service]
    SQI[sqi-feature-service]
    BASE[baseline-engine]
    EVT[event-engine]
    FC[forecast-engine]
    DOC[doc-coding-service]
    RET[retrieval-service]
    LLMGW[llm-gateway]
    POL[policy-engine]
    GOV[governance-service]
    PG[(postgres)]
    QD[(qdrant)]
    RD[(redis)]
    MO[(minio)]
  end
  LLMGW -->|OpenAI-compatible HTTP| VLLM
  GWAPI --> PSE & ING & RET & LLMGW & POL & GOV
  ING --> SQI --> BASE --> EVT --> PSE
  FC --> PSE
  DOC --> PSE
  RET --> QD & PG
  PSE --> PG
  GOV --> PG
  ING --> RD
  DOC --> MO
```

The LLM Gateway reaches host vLLM via `http://host.docker.internal:8000/v1` (default) or the in-Compose `vllm` service (profile `gpu`).

## 2. Run the model on the host

The model server runs on the host and exposes an OpenAI-compatible endpoint; Docker services route to it via `LLM_GATEWAY_BASE_URL`. The serving engine depends on the platform.

### 2a. Production — H200 DGX slice (CUDA / vLLM)
```bash
# Linux host with NVIDIA GPU. fp8 preferred at 80–100 GB; awq is the fallback.
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.6-35B-A3B \
  --quantization fp8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --port 8000
```
Set `LLM_GATEWAY_BASE_URL=http://host.docker.internal:8000/v1` (or the `vllm` container in §4).

### 2b. Local dev/demo — MacBook M5 Pro (Metal)

The model runs **natively on the macOS host** (Metal cannot be reached from inside
Docker) and exposes an OpenAI-compatible endpoint. Two interchangeable ways to serve
the same MLX quant — pick one:

**Option A — LM Studio (GUI, recommended for demo).** Load `mlx-community/Qwen3.6-35B-A3B-4bit`,
then Developer → Local Server → Start on port **1234**. It serves `/v1/*`; the API model
id is under `GET /v1/models` (here: `qwen3.6-35b-a3b`).

**Option B — mlx-lm (headless).**
```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8000   # verify the MoE quant loads first
```

Wire it up (`.env`):
```
LLM_BACKEND=mlx
LLM_GATEWAY_BASE_URL=http://host.docker.internal:1234/v1   # LM Studio (use :8000 for mlx-lm)
PRIMARY_MODEL=qwen3.6-35b-a3b                              # the id from GET /v1/models
```
Docker Desktop reaches the host via `host.docker.internal` (already wired on the
`llm-gateway` service; verified — a container `curl host.docker.internal:1234/v1/models`
returns the model). **A host process** (e.g. `pytest`/scripts run directly on the Mac,
not in a container) must use `http://localhost:1234/v1` instead — `host.docker.internal`
only resolves inside containers. Keep context modest; embeddings/reranker small or CPU.
The `gpu` container profile in §4 does **not** apply on Mac.

## 3. docker-compose.yml (core)

```yaml
name: patient-copilot
networks: { hasa-net: {} }
volumes: { pgdata: {}, qdrant: {}, minio: {}, modelcache: {} }

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: hasa
      POSTGRES_USER: hasa
      POSTGRES_PASSWORD: ${PG_PASSWORD}
    volumes: ["pgdata:/var/lib/postgresql/data"]
    networks: [hasa-net]
    healthcheck: { test: ["CMD-SHELL","pg_isready -U hasa"], interval: 10s, retries: 5 }

  qdrant:
    image: qdrant/qdrant:latest
    volumes: ["qdrant:/qdrant/storage"]
    networks: [hasa-net]

  redis:
    image: redis:7
    networks: [hasa-net]

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment: { MINIO_ROOT_USER: ${MINIO_USER}, MINIO_ROOT_PASSWORD: ${MINIO_PASSWORD} }
    volumes: ["minio:/data"]
    networks: [hasa-net]

  api-gateway:
    build: ./services/api-gateway
    environment:
      DATABASE_URL: postgresql://hasa:${PG_PASSWORD}@postgres:5432/hasa
      JWT_PUBLIC_KEY: ${JWT_PUBLIC_KEY}
    ports: ["8080:8080"]
    depends_on: [postgres, qdrant, redis]
    networks: [hasa-net]

  patient-state-engine: { build: ./services/patient-state-engine, environment: { DATABASE_URL: postgresql://hasa:${PG_PASSWORD}@postgres:5432/hasa }, networks: [hasa-net], depends_on: [postgres] }
  ingestion-service:    { build: ./services/ingestion-service, environment: { DATABASE_URL: postgresql://hasa:${PG_PASSWORD}@postgres:5432/hasa, REDIS_URL: redis://redis:6379 }, networks: [hasa-net], depends_on: [postgres, redis] }
  sqi-feature-service:  { build: ./services/sqi-feature-service, networks: [hasa-net], depends_on: [redis] }
  baseline-engine:      { build: ./services/baseline-engine, environment: { DATABASE_URL: postgresql://hasa:${PG_PASSWORD}@postgres:5432/hasa }, networks: [hasa-net], depends_on: [postgres] }
  event-engine:         { build: ./services/event-engine, environment: { DATABASE_URL: postgresql://hasa:${PG_PASSWORD}@postgres:5432/hasa }, networks: [hasa-net], depends_on: [postgres] }
  forecast-engine:      { build: ./services/forecast-engine, environment: { DATABASE_URL: postgresql://hasa:${PG_PASSWORD}@postgres:5432/hasa }, networks: [hasa-net], depends_on: [postgres] }
  doc-coding-service:   { build: ./services/doc-coding-service, environment: { MINIO_ENDPOINT: minio:9000 }, networks: [hasa-net], depends_on: [minio] }
  retrieval-service:    { build: ./services/retrieval-service, environment: { QDRANT_URL: http://qdrant:6333, DATABASE_URL: postgresql://hasa:${PG_PASSWORD}@postgres:5432/hasa }, networks: [hasa-net], depends_on: [qdrant, postgres] }
  policy-engine:        { build: ./services/policy-engine, environment: { RULESET_VERSION: ${RULESET_VERSION} }, networks: [hasa-net] }
  governance-service:   { build: ./services/governance-service, environment: { DATABASE_URL: postgresql://hasa:${PG_PASSWORD}@postgres:5432/hasa }, networks: [hasa-net], depends_on: [postgres] }

  llm-gateway:
    build: ./services/llm-gateway
    environment:
      LLM_PROFILE: ${LLM_PROFILE:-local}              # local | external_deidentified | dev
      LLM_GATEWAY_BASE_URL: ${LLM_GATEWAY_BASE_URL:-http://host.docker.internal:8000/v1}
      OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}      # only used by external_* profiles
      PRIMARY_MODEL: Qwen/Qwen3.6-35B-A3B
      FALLBACK_MODEL_SLUG: qwen/qwen3.6-35b-a3b
    extra_hosts: ["host.docker.internal:host-gateway"]
    networks: [hasa-net]
```

## 4. Optional: vLLM in a container (profile `gpu`) — **H200 / CUDA only**

Only valid on the Linux + NVIDIA (H200) path. **Not usable on the Mac** — Metal GPUs cannot be passed into containers, so on macOS the model must run on the host (§2b) and this profile is left off.

```yaml
  vllm:
    image: vllm/vllm-openai:latest
    profiles: ["gpu"]                 # H200/CUDA hosts only; do not enable on macOS
    command: >
      --model Qwen/Qwen3.6-35B-A3B --quantization fp8
      --max-model-len 32768 --gpu-memory-utilization 0.90
    volumes: ["modelcache:/root/.cache/huggingface"]
    ports: ["8000:8000"]
    networks: [hasa-net]
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: 1, capabilities: ["gpu"] }]
```
Run with `docker compose --profile gpu up` and set `LLM_GATEWAY_BASE_URL=http://vllm:8000/v1`. On the Mac, omit the profile and point the gateway at the host mlx-lm server (§2b).

## 5. .env (template)

```env
PG_PASSWORD=change-me
MINIO_USER=hasa
MINIO_PASSWORD=change-me
JWT_PUBLIC_KEY=...
LLM_PROFILE=local
LLM_GATEWAY_BASE_URL=http://host.docker.internal:8000/v1
OPENROUTER_API_KEY=          # leave empty in production
RULESET_VERSION=2026.06.0
PRIMARY_MODEL=Qwen/Qwen3.6-35B-A3B
```

## 6. Production hardening (checklist)

- `LLM_PROFILE=local`, `OPENROUTER_API_KEY` empty → no external egress path active.
- Postgres + MinIO volumes on encrypted storage; TLS terminators in front of api-gateway.
- Secrets via Docker secrets / vault, not `.env` in prod.
- Network policy: only `llm-gateway` may reach the host/external; only `api-gateway` is exposed publicly.
- Resource limits and healthchecks on every service; restart policies set.
