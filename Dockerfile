# Dockerfile
#
# Multi-stage build for the churn-prediction project.
# Two targets: "api" (FastAPI) and "streamlit" (dashboard).

FROM python:3.12-slim AS base

WORKDIR /app

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

# FastAPI target 
FROM base AS api

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Streamlit target 
FROM base AS streamlit

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
