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
