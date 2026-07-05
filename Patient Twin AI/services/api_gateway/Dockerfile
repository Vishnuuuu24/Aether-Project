FROM python:3.12-slim

WORKDIR /app

# Build context is the repo ROOT (see docker-compose.yml) so shared packages can be
# vendored in. The gateway imports contracts from `schemas/` and the auth primitives
# from `core/` — it never redefines them (CLAUDE.md hard rule).
COPY services/api_gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/api_gateway/app/ ./app/
COPY schemas/ ./schemas/
COPY core/ ./core/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
