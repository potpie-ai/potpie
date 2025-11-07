# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git procps wget ca-certificates

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install supervisor
RUN apt-get update && apt-get install -y supervisor

# Install Celery
RUN pip install --no-cache-dir celery

# Install NLTK and download required data
RUN pip install --no-cache-dir nltk
RUN python -c "import nltk; nltk.download('punkt');"

# Install gVisor (runsc) for command isolation in K8s/Linux environments
# This allows running isolated commands within the container
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then \
        ARCH="x86_64"; \
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then \
        ARCH="arm64"; \
    else \
        echo "Unsupported architecture: $ARCH, skipping gVisor installation"; \
        exit 0; \
    fi && \
    URL=https://storage.googleapis.com/gvisor/releases/release/latest/${ARCH} && \
    wget -q ${URL}/runsc ${URL}/runsc.sha512 && \
    sha512sum -c runsc.sha512 && \
    chmod a+rx runsc && \
    mv runsc /usr/local/bin/runsc && \
    rm -f runsc.sha512 || echo "gVisor installation failed, continuing without it"

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
