# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git procps wget curl gnupg2 ca-certificates supervisor build-essential && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain for local PyO3/maturin path dependencies
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# Set the working directory in the container
WORKDIR /app

# Ensure Rust tooling is available while building Python dependencies
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /bin/

# Copy dependency metadata and local path packages (editable deps in uv.lock)
COPY pyproject.toml uv.lock ./
COPY app/src/context-engine ./app/src/context-engine
COPY app/src/integrations ./app/src/integrations
COPY app/src/parsing ./app/src/parsing
COPY app/src/sandbox ./app/src/sandbox

# Install dependency layers first, including the local Rust extension
RUN uv sync --frozen --no-install-project

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

# Install the project after the full source tree is available
RUN uv sync --frozen

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
