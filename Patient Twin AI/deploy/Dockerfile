# Shared runtime image for the app services (docs/08 §6; docs/15 T7.3).
#
# One image, config-switched: this is a monorepo where services import the shared
# packages (schemas/, core/, ai/) and each other, so every service needs the same
# vendored tree. The concrete service is selected by the compose `command`
# (`uvicorn services.<svc>.app.main:app`) — where things run is configuration, not a
# forked image (CLAUDE.md). The api-gateway keeps its own lean Dockerfile (it needs
# only core/ + schemas/).
FROM python:3.12-slim

WORKDIR /app

# Build context is the repo ROOT (see docker-compose.yml).
COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Vendor the shared contracts + libraries + services + clinical config stubs.
COPY schemas/ ./schemas/
COPY core/ ./core/
COPY ai/ ./ai/
COPY services/ ./services/
COPY config/ ./config/

EXPOSE 8000

# Overridden per service by the compose `command`. Fail loudly if run bare.
CMD ["python", "-c", "raise SystemExit('set a uvicorn service target via the compose command')"]
