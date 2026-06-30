FROM python:3.12-slim

WORKDIR /app

# Build context is the repo ROOT (see docker-compose.yml) so the shared
# `schemas/` package can be vendored in. Services import contracts from
# schemas/ — they never redefine them (CLAUDE.md hard rule).
COPY services/api-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api-gateway/app/ ./app/
COPY schemas/ ./schemas/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
