#!/bin/bash

# Build the Docker image
docker build -t fastapi-mongo-proxy .

# Run the Docker container in detached mode
docker run -d --name fastapi-mongo-proxy -p 8000:8000 fastapi-mongo-proxy
