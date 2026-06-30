# 09 — Repository Blueprint

Monorepo. Python 3.12. Each service is independently buildable but shares the `schemas` and `core` packages so contracts cannot drift.

```text
patient-copilot/
├── README.md
├── docker-compose.yml
├── .env.example
├── pyproject.toml                 # workspace / shared tooling (ruff, mypy, pytest)
├── Makefile                       # up, down, test, lint, migrate, seed, replay
│
├── schemas/                       # SINGLE SOURCE OF CONTRACTS (imported everywhere)
│   ├── reading.py                 # per-reading schema (04 §2)
│   ├── psg.py                     # PSG nodes/edges + projection (04 §3, §5)
│   ├── output_contract.py         # output contract (04 §6)
│   ├── fhir/                      # FHIR resource models + OMOP mirror mapping
│   └── openapi/                   # generated OpenAPI specs per service
│
├── core/                          # shared libs (no business decisions here)
│   ├── auth/                      # JWT, RBAC, consent enforcement
│   ├── audit/                     # hash-chained audit writer (04 §7)
│   ├── versioning/                # model/ruleset/prompt/schema version registry client
│   ├── db/                        # SQLAlchemy models, migrations (alembic)
│   └── telemetry/                 # OpenTelemetry helpers
│
├── services/
│   ├── api-gateway/               # auth, routing, rate limit (07)
│   ├── ingestion-service/         # adapters + normalisation (07 §3)
│   │   └── adapters/              # healthkit/ health_connect/ fitbit/ replay/ csv
│   ├── sqi-feature-service/       # SQI + classical features (FeatureExtractor)
│   ├── baseline-engine/           # StatisticalBaselineEngine (05) + Event Engine
│   ├── event-engine/              # co-occurrence/persistence rules (05 §6)
│   ├── forecast-engine/           # Forecaster (05 §7)
│   ├── doc-coding-service/        # Docling/Marker OCR + MedCAT coding (04 §4)
│   ├── retrieval-service/         # BM25 + dense (MedCPT/BGE) + reranker
│   ├── llm-gateway/               # LLMGateway: local vLLM ↔ OpenRouter profiles (06 §6)
│   ├── policy-engine/             # deterministic Policy Engine (06)
│   ├── governance-service/        # consent, audit query, outcome capture, versions
│   └── patient-state-engine/      # owns the PSG; validate+commit; projection builder
│
├── ai/
│   ├── prompts/                   # versioned prompt templates (no secrets, no PHI)
│   ├── interfaces/                # BaselineEngine, FeatureExtractor, Retriever, Forecaster, LLMGateway
│   ├── baseline/                  # statistical impl; DEFERRED foundation-encoder impl stub
│   ├── features/                  # classical extractors; DEFERRED PaPaGei-S/Pulse-PPG stub
│   └── retrieval/                 # embedders, reranker wiring
│
├── models/                        # model cards, quant configs, NOT weights (weights via cache/registry)
│
├── datasets/                      # replay manifests for PPG-DaLiA, WESAD, MESA, SHHS, WildPPG
│   └── replay/                    # harness config (07 §3 /ingest/replay)
│
├── infrastructure/
│   ├── docker/                    # per-service Dockerfiles
│   ├── migrations/                # alembic
│   └── seed/                      # KB ingestion + demo patient seed
│
├── evaluation/                    # harnesses + metrics (11)
│   ├── retrieval/                 # Recall@K, MRR, nDCG
│   ├── baseline/                  # deviation P/R, calibration/ECE
│   ├── forecast/                  # MAE/RMSE
│   ├── safety/                    # hallucination, grounding, abstention, scope-violation
│   └── datasets/                  # offline clinical eval fixtures
│
└── docs/                          # THIS handoff package (00–11)
```

## Conventions

- **Contracts live in `schemas/` only.** Services import them; no service redefines a contract locally.
- **Interfaces live in `ai/interfaces/`.** Deferred components (foundation encoder, FM forecaster) are new implementations of these, never new call sites.
- **Every service** exposes `/healthz`, `/readyz`, `/metrics`, emits audit events via `core/audit`, and stamps `versions` on every output.
- **Migrations** are forward-only and backward-compatible (schemas never break consumers).
- **No PHI** in `ai/prompts/`, logs, or anything reachable by an `external_*` LLM profile.
- `Makefile` targets: `make up` (compose), `make migrate`, `make seed`, `make replay DATASET=PPG-DaLiA`, `make test`, `make eval`.
