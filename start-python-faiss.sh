#!/bin/bash

# Set default Python backend port
PYTHON_PORT=5001

# Check if the port is already in use and get PID if it is
PYTHON_PID=$(lsof -t -i:$PYTHON_PORT 2>/dev/null)

if [ -n "$PYTHON_PID" ]; then
    echo "Port $PYTHON_PORT is already in use by process $PYTHON_PID"
    echo "Checking if it's our Python backend..."
    
    # Check if the process is Python
    if ps -p $PYTHON_PID -o command= | grep -q "python"; then
        echo "Existing Python process found on port $PYTHON_PORT. Reusing it."
        exit 0
    else
        echo "Port $PYTHON_PORT is used by a non-Python process. Please free this port and try again."
        exit 1
    fi
fi

# Check Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed or not in PATH"
    exit 1
fi

# Check required packages
REQUIRED_PACKAGES=(fastapi uvicorn pydantic python-multipart openai)
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $package" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo "Missing Python packages: ${MISSING_PACKAGES[*]}"
    echo "Installing missing packages..."
    pip install "${MISSING_PACKAGES[@]}" --quiet
fi

# Check if faiss-cpu is installed
if ! python -c "import faiss" &> /dev/null; then
    echo "FAISS not installed, installing it now..."
    pip install faiss-cpu
fi

# Check LangChain and necessary packages
if ! python -c "import langchain, langchain_community.vectorstores.faiss" &> /dev/null; then
    echo "LangChain packages not installed, installing them now..."
    pip install langchain langchain-community
fi

# Start Python backend
echo "Starting Python backend on port $PYTHON_PORT..."
cd "$(dirname "$0")"
python python_backend/run_faiss.py &

# Store PID of Python process
echo $! > python_backend.pid

# Wait for backend to start 
echo "Waiting for Python backend to start..."
sleep 2

# Check if the server started successfully
if ! lsof -i:$PYTHON_PORT > /dev/null; then
    echo "Failed to start Python backend"
    exit 1
fi

echo "Python backend with FAISS started successfully"