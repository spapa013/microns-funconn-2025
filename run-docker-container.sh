#!/bin/bash
# Auto-generate .env file with registry name based on git remote

# Generate or update the .env file with the correct registry name
cd "$(dirname "$0")"
REGISTRY=$(./show-registry.sh)

if [ ! -f .env ] || ! grep -q "^REGISTRY_NAME=" .env; then
  # Either .env file doesn't exist or it doesn't contain REGISTRY_NAME
  echo "REGISTRY_NAME=$REGISTRY" >> .env
else
  # REGISTRY_NAME already exists in .env, do nothing
  echo "REGISTRY_NAME already set in .env file, keeping existing value"
fi

echo "Updated .env file with registry name from git remote"
echo "Current .env file content:"
cat .env

echo -e "\nRunning docker-compose with these settings..."
exec docker compose up -d
