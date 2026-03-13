# Use minimal Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    supervisor \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install uv (faster dependency manager)
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /bin/

# Copy only dependency files first (for caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-cache

# Set virtual environment path
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Download required NLTK data
RUN uv run python -c "import nltk; nltk.download('punkt')"

# Copy application code
COPY . .

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set New Relic config path
ENV NEW_RELIC_CONFIG_FILE=/app/newrelic.ini

# Create non-root user (IMPORTANT)
RUN useradd -m appuser && chown -R appuser:appuser /app

USER appuser

# Expose application port
EXPOSE 8001

# Healthcheck (important for reliability)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Start application via supervisor
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
