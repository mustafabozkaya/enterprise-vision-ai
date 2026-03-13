# Docker Container Optimization Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `docker-compose.yml` for one-command FastAPI+Streamlit startup, split Dockerfile into separate `api` and `ui` build targets, generate a `uv.lock` for reproducible builds, and fix missing dependencies in `pyproject.toml`.

**Architecture:** The current Dockerfile only runs Streamlit; FastAPI has no container. We add separate named build targets (`api` on port 8000, `ui` on port 8501) both inheriting from a shared `builder` stage. Since both services require `ultralytics` for inference, the shared builder installs the full dependency set — no size reduction is claimed, but services are isolated. `uv.lock` is generated from `pyproject.toml` after syncing its deps with `requirements.txt`; the Dockerfile then uses `uv sync --frozen` for bit-identical installs.

**Tech Stack:** Docker BuildKit, uv (lock file + sync), docker-compose v2, FastAPI, Streamlit, python:3.11-slim-bookworm

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `Dockerfile` | Modify | Add `api` and `ui` named build targets |
| `pyproject.toml` | Modify | Sync `[project].dependencies` with `requirements.txt` |
| `uv.lock` | Create | Reproducible lock file (via `uv lock` after pyproject sync) |
| `docker-compose.yml` | Create | Orchestrate `api` + `ui` services |
| `.github/workflows/ci.yml` | Modify | Build both targets; extend change-detection to `uv.lock` and `docker-compose.yml` |

Note: `requirements.txt` stays unchanged as the full local-dev union. `requirements-api.txt` / `requirements-ui.txt` are NOT created — since ultralytics is required by both services, splitting provides no meaningful size reduction and adds maintenance overhead.

---

## Chunk 1: Sync pyproject.toml and generate uv.lock

### Task 1: Align pyproject.toml dependencies with requirements.txt

**Files:**
- Modify: `pyproject.toml`

Current `pyproject.toml` `[project].dependencies` is missing these packages that are in `requirements.txt`:
- `pydantic>=2.5.0`
- `pydantic-settings>=2.1.0`
- `python-dotenv>=1.0.0`
- `httpx>=0.26.0`
- `huggingface_hub>=0.20.0`
- `python-dateutil>=2.8.2`

It also still lists `streamlit-webrtc>=0.47.0` and `av>=12.0.0` which add ~200MB of C extensions for RTSP — these can be removed since the RTSP live-stream feature is not implemented.

- [ ] **Step 1: Update `[project].dependencies` in `pyproject.toml`**

Find the `dependencies = [` list in `pyproject.toml` and replace it with:

```toml
dependencies = [
    # UI
    "streamlit>=1.28.0",
    # Vision inference
    "ultralytics>=8.0.0",
    "opencv-python-headless>=4.8.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pillow>=10.0.0",
    "plotly>=5.15.0",
    # FastAPI
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "python-multipart>=0.0.6",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "slowapi>=0.1.9",
    "python-dotenv>=1.0.0",
    # HTTP / files
    "aiofiles>=23.2.1",
    "httpx>=0.26.0",
    "huggingface_hub>=0.20.0",
    "python-dateutil>=2.8.2",
]
```

- [ ] **Step 2: Verify uv can resolve the new deps**

```bash
cd C:/Users/kurtar/Desktop/computervision_projects
uv lock
```
Expected: `uv.lock` created or updated. No resolution errors.

- [ ] **Step 3: Verify lock file was created and is non-empty**

```bash
ls -lh uv.lock
```
Expected: file exists, size > 10KB.

- [ ] **Step 4: Verify environment installs from lock**

```bash
uv sync
uv run pytest tests/ -q
```
Expected: `29 passed`

- [ ] **Step 5: Verify uv.lock is not gitignored**

```bash
grep "uv.lock" .gitignore
```
Expected: no output (uv.lock should be committed, not ignored)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: sync pyproject deps with requirements.txt and add uv.lock"
```

---

## Chunk 2: Multi-Target Dockerfile

### Task 2: Rewrite Dockerfile with api and ui build targets

**Files:**
- Modify: `Dockerfile`

Current Dockerfile: `builder` → `production` (Streamlit only, port 8501).
New Dockerfile: `builder` → `api` (FastAPI, port 8000) and `builder` → `ui` (Streamlit, port 8501).

The builder uses `uv sync --frozen` which reads `uv.lock` for 100% reproducible installs.

- [ ] **Step 1: Replace `Dockerfile` with the following**

```dockerfile
# =============================================================================
# Enterprise Vision AI - Multi-target Dockerfile
# Targets: api (FastAPI on 8000), ui (Streamlit on 8501)
#
# Build individual targets:
#   docker build --target api -t vision-api .
#   docker build --target ui  -t vision-ui  .
#
# Or use docker-compose (recommended):
#   docker compose up --build
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder — install ALL dependencies via uv.lock (reproducible)
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Build-time system dependencies needed to compile native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.6.6

# Create virtual environment
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy dependency manifests first (maximises layer cache hits)
COPY pyproject.toml uv.lock ./

# Install from lock file — bit-identical across all builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Install the package itself (no deps, already installed above)
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python -e . --no-deps

# -----------------------------------------------------------------------------
# Stage 2: api — FastAPI service (port 8000)
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS api

WORKDIR /app

# Runtime system deps for vision inference (OpenCV + YOLO)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser pyproject.toml ./

RUN mkdir -p /app/uploads /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "enterprise_vision_ai.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 3: ui — Streamlit frontend (port 8501)
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS ui

WORKDIR /app

# Runtime system deps for Streamlit rendering + OpenCV display
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser pages/ pages/
COPY --chown=appuser:appuser services/ services/
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser pyproject.toml ./

RUN mkdir -p /app/uploads /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

- [ ] **Step 2: Verify FastAPI `/health` route exists**

```bash
grep -n "health\|/health" src/enterprise_vision_ai/api/main.py | head -10
```
Expected: a route decorated with `@app.get("/health")` or included via the health router.

- [ ] **Step 3: Build `api` target locally**

```bash
cd C:/Users/kurtar/Desktop/computervision_projects
DOCKER_BUILDKIT=1 docker build --target api -t vision-api:local .
```
Expected: build succeeds.
```bash
docker image ls vision-api:local
```
Note the size.

- [ ] **Step 4: Build `ui` target locally**

```bash
DOCKER_BUILDKIT=1 docker build --target ui -t vision-ui:local .
```
Expected: build succeeds.
```bash
docker image ls vision-ui:local
```

- [ ] **Step 5: Commit**

```bash
git add Dockerfile
git commit -m "feat: add api and ui build targets to Dockerfile, use uv sync --frozen"
```

---

## Chunk 3: docker-compose.yml

### Task 3: Create docker-compose.yml for one-command startup

**Files:**
- Create: `docker-compose.yml`

Note: A broken `docker-compose.yml` may already exist referencing a non-existent `development` target. This task overwrites it.

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
# Enterprise Vision AI - Docker Compose
# Usage:
#   docker compose up --build        # build and start both services
#   docker compose up                # start (images already built)
#   docker compose up api            # start FastAPI only
#   docker compose up ui             # start Streamlit only
#   docker compose down              # stop all
#   docker compose logs -f api       # tail FastAPI logs

services:
  api:
    build:
      context: .
      target: api
    image: vision-api:latest
    container_name: vision-api
    ports:
      - "8000:8000"
    volumes:
      - uploads:/app/uploads
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      start_period: 15s
      retries: 3

  ui:
    build:
      context: .
      target: ui
    image: vision-ui:latest
    container_name: vision-ui
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
      - PYTHONUNBUFFERED=1
    depends_on:
      api:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"]
      interval: 30s
      timeout: 10s
      start_period: 20s
      retries: 3

volumes:
  uploads:
    driver: local
```

`API_URL=http://api:8000` passes the internal Docker network hostname to Streamlit so `get_api_url()` in `app.py` returns it instead of scanning localhost ports.

- [ ] **Step 2: Validate compose file**

```bash
docker compose config
```
Expected: full YAML printed without errors.

- [ ] **Step 3: Start both services**

```bash
docker compose up --build -d
```
Expected:
```
 ✔ Container vision-api  Healthy
 ✔ Container vision-ui   Started
```

- [ ] **Step 4: Verify API**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/models
```
Expected: both return JSON with HTTP 200.

- [ ] **Step 5: Verify UI shows API as active**

Open `http://localhost:8501` in browser.
Expected: sidebar shows `🟢 Aktif`.

- [ ] **Step 6: Stop**

```bash
docker compose down
```

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add docker-compose for one-command api+ui startup"
```

---

## Chunk 4: CI Updates

### Task 4: Update CI to build both targets and extend change-detection

**Files:**
- Modify: `.github/workflows/ci.yml`

Two fixes needed:
1. Build both `api` and `ui` targets (not just one build)
2. Extend change-detection pattern to include `uv.lock` and `docker-compose.yml`

- [ ] **Step 1: Find the docker job and change-detection pattern**

```bash
grep -n "docker_changes\|changed=\|Dockerfile\|target:" .github/workflows/ci.yml | head -20
```

- [ ] **Step 2: Update the change-detection `run:` block**

Find the step that sets `changed=true/false`. Update the grep pattern:

```bash
# Before:
if echo "$CHANGED" | grep -qE "^(Dockerfile|requirements.*\.txt|pyproject\.toml|src/)"; then

# After:
if echo "$CHANGED" | grep -qE "^(Dockerfile|docker-compose.*\.yml|uv\.lock|requirements.*\.txt|pyproject\.toml|src/)"; then
```

- [ ] **Step 3: Update the docker build step to build both targets**

Find the `Build Docker image` step. Replace the single build with two sequential builds:

```yaml
      - name: Build api target
        if: steps.docker_changes.outputs.changed == 'true'
        uses: docker/build-push-action@v5
        with:
          context: .
          target: api
          push: false
          cache-from: type=gha
          cache-to: type=gha,mode=max
          tags: vision-api:ci

      - name: Build ui target
        if: steps.docker_changes.outputs.changed == 'true'
        uses: docker/build-push-action@v5
        with:
          context: .
          target: ui
          push: false
          cache-from: type=gha
          cache-to: type=gha,mode=max
          tags: vision-ui:ci
```

- [ ] **Step 4: Commit and push, verify CI passes**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: build api and ui targets, extend docker change-detection"
git push origin main
```

Monitor: `gh run watch`
Expected: docker job builds both targets.

---

## Success Criteria

- [ ] `uv.lock` committed and `uv sync --frozen` works from scratch
- [ ] `docker compose up --build` starts both FastAPI (`:8000`) and Streamlit (`:8501`)
- [ ] `curl http://localhost:8000/health` returns 200 from containerized API
- [ ] `curl http://localhost:8000/api/v1/models` returns 3 models, no 500
- [ ] Streamlit sidebar shows `🟢 Aktif` (reads `API_URL=http://api:8000`)
- [ ] CI builds both `api` and `ui` Docker targets without failure
- [ ] `docker compose down` cleans up both containers cleanly
