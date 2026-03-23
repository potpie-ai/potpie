# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies (ripgrep: agent bash_command whitelist; xz-utils: extract colgrep)
RUN apt-get update && apt-get install -y \
    git procps wget curl gnupg2 ca-certificates supervisor ripgrep xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ColGREP CLI (semantic search); indices use XDG_DATA_HOME under REPOS_BASE_PATH (see colgrep_index).
# Prebuilt Linux asset is x86_64-only in upstream releases; skip on other architectures.
ARG COLGREP_VERSION=1.1.3
RUN set -eux; \
    ARCH="$(dpkg --print-architecture)"; \
    if [ "$ARCH" = "amd64" ]; then \
      curl -fsSL -o /tmp/colgrep.tar.xz \
        "https://github.com/lightonai/next-plaid/releases/download/v${COLGREP_VERSION}/colgrep-x86_64-unknown-linux-gnu.tar.xz"; \
      tar -xJf /tmp/colgrep.tar.xz -C /tmp; \
      mv "/tmp/colgrep-x86_64-unknown-linux-gnu/colgrep" /usr/local/bin/colgrep; \
      rm -rf /tmp/colgrep-x86_64-unknown-linux-gnu; \
      chmod +x /usr/local/bin/colgrep; \
      rm -f /tmp/colgrep.tar.xz; \
      colgrep --version; \
    else \
      echo "Skipping ColGREP install: no prebuilt binary for $ARCH (install manually or use amd64 image)"; \
    fi

# Set the working directory in the container
WORKDIR /app

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /bin/

# Copy dependency metadata first for better layer caching
COPY pyproject.toml uv.lock ./

# Install project dependencies using uv (creates .venv)
RUN uv sync --frozen --no-cache

# Ensure the virtual environment binaries are on PATH
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Download required NLTK data inside the managed environment
RUN uv run python -c "import nltk; nltk.download('punkt')"

# Install gVisor (runsc) via official APT repository for command isolation
RUN curl -fsSL https://gvisor.dev/archive.key | gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" > /etc/apt/sources.list.d/gvisor.list && \
    apt-get update && \
    apt-get install -y runsc || echo "gVisor runsc package not available for this architecture; continuing without it" && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code into the container
COPY . .

# env path for newrelic.ini
ENV NEW_RELIC_CONFIG_FILE=/app/newrelic.ini

# Copy the Supervisor configuration file into the container
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose the port that the app runs on
EXPOSE 8001

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run Supervisor when the container launches
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
