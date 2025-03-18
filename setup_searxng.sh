#!/bin/bash

# Set port and directories
PORT=8080
CURRENT_DIR="$(pwd)"
INSTANCE_DIR="${CURRENT_DIR}/searxng"

# Create instance directory if it doesn't exist
if [ ! -d "$INSTANCE_DIR" ]; then
    echo "Creating SearXNG instance directory..."
    mkdir -p "$INSTANCE_DIR"
fi

# Copy pre-configured settings file if it exists
if [ -f "${CURRENT_DIR}/default_searxng.yml" ]; then
    echo "Using pre-configured settings file..."
    cp "${CURRENT_DIR}/default_searxng.yml" "${INSTANCE_DIR}/settings.yml"
    echo "✅ Custom configuration copied!"
else
    echo "⚠️ No custom settings file found (default_searxng.yml). Will use default settings."
fi

# Pull the latest SearXNG Docker image
echo "Pulling the latest SearXNG Docker image..."
docker pull searxng/searxng

# Stop existing container if running
echo "Stopping any existing SearXNG container..."
docker stop searxng_instance 2>/dev/null || true

# Start SearXNG container with volume mount to current directory
echo "Starting SearXNG on port $PORT..."
docker run --rm \
       -d -p ${PORT}:8080 \
       -v "${INSTANCE_DIR}:/etc/searxng" \
       -e "BASE_URL=http://localhost:$PORT/" \
       -e "INSTANCE_NAME=my-instance" \
       --name searxng_instance \
       searxng/searxng

echo "✅ SearXNG is running at: http://localhost:$PORT"
