#!/bin/bash
# Test script to build and verify the funconnect Docker image

echo "Building Docker image using docker-compose-local.yml..."
docker compose -f docker-compose-local.yml build

echo "Starting the test container..."
docker compose -f docker-compose-local.yml up -d test

echo "Waiting for container to be ready..."
sleep 5  # Give the container a moment to initialize

echo "Verifying funconnect installation..."
docker compose -f docker-compose-local.yml exec test python -c "import funconnect; print('funconnect version:', funconnect.__version__)"

echo "Stopping the container..."
docker compose -f docker-compose-local.yml down

echo "Done! The funconnect package is installed in the Docker image."