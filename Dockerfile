# =============================================================================
# Multi-stage Dockerfile for Macro Quant Trading System
# =============================================================================

# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies (frozen if lock file exists)
RUN uv sync --no-dev --no-install-project 2>/dev/null || uv sync --no-dev

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mquser && useradd -r -g mquser -d /app mquser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src/ /app/src/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini

# Set environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER mquser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

# Default command: run the scheduler
CMD ["python", "-m", "src.scheduler.jobs"]
