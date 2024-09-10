# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install Git and PostgreSQL development libraries
RUN apt-get update && apt-get install -y git procps

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK and download required data
RUN pip install --no-cache-dir nltk
RUN python -c "import nltk; nltk.download('punkt');"

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app runs on
EXPOSE 8001

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run Alembic migrations before starting the application
CMD ["sh", "-c", "WORKERS=$(( $(nproc) )); echo 'Starting Gunicorn with' $WORKERS 'workers'; alembic upgrade head && gunicorn --workers $WORKERS --worker-class uvicorn.workers.UvicornWorker --timeout 1800 --bind 0.0.0.0:8001 --log-level debug app.main:app"]
