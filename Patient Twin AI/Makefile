# Load local .env (gitignored) so host-run targets (migrate, test) see PG_PASSWORD
# etc. `docker compose` already auto-loads .env; this covers the non-compose targets.
ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: up down logs ps migrate seed replay test lint fmt eval

up:
	docker compose up -d --build
	@echo "Waiting for services..."
	@sleep 3
	@curl -sf http://localhost:8080/readyz || echo "NOT READY YET — check: make logs"

up-gpu:
	docker compose --profile gpu up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

migrate:
	cd core/db && alembic upgrade head

seed:
	python3 infrastructure/seed/seed_demo_patient.py

replay:
	python3 -m services.ingestion_service.adapters.replay --dataset $(DATASET)

test:
	pytest -q

lint:
	ruff check .
	mypy schemas core services

fmt:
	ruff format .

eval:
	python3 -m evaluation.run_all
