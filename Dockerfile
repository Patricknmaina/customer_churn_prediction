# Dockerfile
#
# Multi-stage build for the churn-prediction project.
# Two targets: "api" (FastAPI) and "streamlit" (dashboard).

FROM python:3.12-slim AS base

WORKDIR /app

# Runtime dependency for xgboost/shap native extensions on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency manifests first (cache-friendly layer)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev, frozen lockfile)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY scripts/ scripts/
COPY api/ api/
COPY streamlit/ streamlit/
COPY artifacts/ artifacts/

# Streamlit target 
FROM base AS streamlit

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# FastAPI target (kept as final stage so platforms that build the default
# final image stage deploy the API service by default)
FROM base AS api

EXPOSE 8000

CMD ["sh", "-c", "uv run uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
