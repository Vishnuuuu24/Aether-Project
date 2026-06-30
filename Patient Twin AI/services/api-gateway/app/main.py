"""api-gateway skeleton. Sprint 0 (docs/10 T0.3): exists only to prove the
stack is wired — /healthz and /readyz green. Real routing (docs/07) is
built in Sprint 4.
"""
from __future__ import annotations

import os

import asyncpg
from fastapi import FastAPI, Response, status
from qdrant_client import QdrantClient

app = FastAPI(title="patient-copilot-api-gateway", version="0.0.1")


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/readyz")
async def readyz(response: Response) -> dict[str, str | bool]:
    """Checks that Postgres and Qdrant are reachable. Used by docker
    healthchecks and by the build agent to confirm `make up` succeeded.
    """
    checks: dict[str, bool] = {}

    db_url = os.environ.get("DATABASE_URL", "")
    try:
        conn = await asyncpg.connect(db_url, timeout=3)
        await conn.execute("SELECT 1")
        await conn.close()
        checks["postgres"] = True
    except Exception:
        checks["postgres"] = False

    try:
        qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
        host = qdrant_url.split("://")[-1].split(":")[0]
        port = int(qdrant_url.split(":")[-1])
        client = QdrantClient(host=host, port=port, timeout=3)
        client.get_collections()
        checks["qdrant"] = True
    except Exception:
        checks["qdrant"] = False

    all_ok = all(checks.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": all_ok, **checks}
