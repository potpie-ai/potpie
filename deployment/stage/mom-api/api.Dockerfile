# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    procps \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir \
    -r requirements.txt \
    celery \
    flower \
    nltk

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt');"

# Copy application code
COPY . .

# Set environment variables
ENV NEW_RELIC_CONFIG_FILE=/app/newrelic.ini
ENV PYTHONUNBUFFERED=1

# Copy the Supervisor configuration file
COPY deployment/stage/mom-api/mom-api-supervisord.conf /etc/supervisor/conf.d/mom-api-supervisord.conf

# Expose ports
EXPOSE 8001 5555

# Add build arguments for commit tracking
ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

# Run Supervisor when the container launches
CMD ["supervisord", "-n", "-c", "/etc/supervisor/conf.d/mom-api-supervisord.conf"]