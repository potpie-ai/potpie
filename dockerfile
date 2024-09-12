# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git procps

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Celery and Flower
RUN pip install --no-cache-dir celery flower

# Install NLTK and download required data
RUN pip install --no-cache-dir nltk
RUN python -c "import nltk; nltk.download('punkt');"

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app runs on
EXPOSE 8001

# Expose the port for Flower
EXPOSE 5555

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Define the command to run multiple services
CMD ["sh", "-c", "\
    set -e; \
    WORKERS=$(( $(nproc) )); \
    echo 'Running database migrations...'; \
    alembic upgrade head; \
    echo 'Starting Celery worker...'; \
    celery -A app.core.celery_config.celery_app worker --loglevel=info -E & \
    CELERY_PID=$!; \
    echo 'Starting Flower...'; \
    flower -A app.core.celery_config.celery_app --port=5555 --broker=redis://redis:6379/0 & \
    FLOWER_PID=$!; \
    echo 'Starting momentum application...'; \
    gunicorn --workers $WORKERS --worker-class uvicorn.workers.UvicornWorker --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app & \
    GUNICORN_PID=$!; \
    wait $CELERY_PID $FLOWER_PID $GUNICORN_PID"]
