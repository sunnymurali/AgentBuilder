#!/bin/bash

# Enable debugging
set -x

# Ensure necessary directories exist
mkdir -p data
mkdir -p uploads

# Check for required Python packages
echo "Checking for required Python packages..."

# Check for Python itself
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Check for ChromaDB installation
python3 -c "import chromadb" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "WARNING: ChromaDB is not installed. The server will run in fallback mode."
  echo "To install ChromaDB, run: pip install chromadb"
  echo "Running in fallback mode..."
else
  echo "ChromaDB is installed. Running with full functionality."
fi

# Check for FastAPI
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "WARNING: FastAPI is not installed. The server may not run correctly."
  echo "To install FastAPI, run: pip install fastapi uvicorn"
  exit 1
fi

# Check for other required packages
for pkg in "pydantic" "uvicorn" "pypdf2" "python-docx" "pandas" "openai" "langchain"
do
  python3 -c "import ${pkg//-/_}" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "WARNING: $pkg is not installed. Some features may not work."
  else
    echo "$pkg is installed."
  fi
done

# Check if port 5001 is already in use
if command -v nc &> /dev/null; then
  if nc -z localhost 5001 2>/dev/null; then
    echo "ERROR: Port 5001 is already in use!"
    echo "Stop any existing Python backend servers before running this script."
    exit 1
  fi
fi

echo "Starting Python backend server on port 5001..."
echo "Working directory: $(pwd)"

# Make sure python_backend directory exists and has app.py
if [ ! -d "python_backend" ]; then
  echo "ERROR: python_backend directory not found!"
  exit 1
fi

if [ ! -f "python_backend/app.py" ]; then
  echo "ERROR: python_backend/app.py not found!"
  exit 1
fi

cd python_backend
echo "Starting uvicorn server..."
python3 -m uvicorn app:app --host 0.0.0.0 --port 5001 --reload