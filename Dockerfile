# T4DM REST API Server
# Multi-stage build for minimal production image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with API dependencies
RUN pip install --no-cache-dir -e ".[api]"


# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r t4dm && useradd -r -g t4dm t4dm

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml ./

# Create directories for models and data
RUN mkdir -p /app/models /app/data && \
    chown -R t4dm:t4dm /app

# Switch to non-root user
USER t4dm

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    T4DM_API_HOST=0.0.0.0 \
    T4DM_API_PORT=8765 \
    T4DM_EMBEDDING_CACHE_DIR=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${T4DM_API_PORT}/api/v1/health || exit 1

# Expose port
EXPOSE 8765

# Run the API server
CMD ["python", "-m", "ww.api.server"]
