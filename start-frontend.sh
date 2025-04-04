#!/bin/bash

# Enable debugging
set -x

# Ensure the necessary directories exist
mkdir -p uploads
mkdir -p data

echo "Starting only the Node.js frontend server..."
echo "Make sure your Python backend is running on port 5001"

# Check if port 5000 is already in use
if command -v nc &> /dev/null; then
  if nc -z localhost 5000 2>/dev/null; then
    echo "ERROR: Port 5000 is already in use!"
    echo "Stop any existing Node.js servers before running this script."
    echo "If you're using Replit, make sure no other workflows are running."
    exit 1
  fi
fi

# Set environment variable to skip Python backend
export SKIP_PYTHON_BACKEND=true

# Check NODE_PATH
echo "Node path: $(which node)"
echo "NPM path: $(which npm)"
echo "Working directory: $(pwd)"

# List package.json to verify it exists
if [ ! -f "package.json" ]; then
  echo "ERROR: package.json not found!"
  exit 1
fi

ls -la package.json

# Start the server
echo "Starting Node.js server with SKIP_PYTHON_BACKEND=true..."
npm run dev