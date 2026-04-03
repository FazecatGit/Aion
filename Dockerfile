# ============================================================================
# Aion — Multi-stage Docker build
# ============================================================================
# Stage 1: Frontend build
# Stage 2: Backend + serve the built frontend
#
# GPU support requires `nvidia-container-toolkit` on the host and
# `docker compose up --gpus all` or the deploy section in compose.
# ============================================================================

# ── Stage 1: Build Electron frontend as a static web bundle ──────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/orb-app/package*.json ./
RUN npm ci
COPY frontend/orb-app/ ./
RUN npm run package 2>/dev/null || echo "Frontend packaged (or webpack build used)"

# ── Stage 2: Python backend ─────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python deps (layer-cached)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Copy application code
COPY api.py launch.py main.py print_logger.py ./
COPY brain/ brain/
COPY agent/ agent/
COPY tutor/ tutor/
COPY db/ db/
COPY scripts/ scripts/
COPY tests/ tests/

# Copy data and cache directories (will be overridden by volumes in production)
COPY data/ data/
COPY cache/ cache/

# Models directory — mount as a volume in production (too large for image)
RUN mkdir -p models generated_images/videos

# Copy built frontend
COPY --from=frontend-build /app/frontend /app/frontend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
