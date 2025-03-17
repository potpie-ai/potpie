# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git procps

# Install uv
RUN pip install uv

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN uv pip install --no-cache-dir -r requirements.txt

# Install supervisor
RUN apt-get update && apt-get install -y supervisor

# Install Celery (already included if in requirements.txt)
RUN uv pip install --no-cache-dir celery

# Install NLTK and download required data
RUN uv pip install --no-cache-dir nltk
RUN python -c "import nltk; nltk.download('punkt');"

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
