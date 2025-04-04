#!/bin/bash

# Enable debugging
set -x

# Ensure the necessary directories exist
mkdir -p uploads
mkdir -p data

# Parse command line options
SKIP_PYTHON=false
for arg in "$@"
do
    case $arg in
        --skip-python)
        SKIP_PYTHON=true
        shift
        ;;
        *)
        # Unknown option
        ;;
    esac
done

# Display environment information
echo "Node version: $(node --version)"
echo "NPM version: $(npm --version)"
echo "Working directory: $(pwd)"

# Check if a Python server is already running on port 5001
python_running=false
if command -v nc &> /dev/null; then
  if nc -z localhost 5001 2>/dev/null; then
    echo "A server is already running on port 5001."
    echo "Assuming this is your Python backend."
    python_running=true
    SKIP_PYTHON=true
  else
    echo "No server detected on port 5001."
  fi
else
  echo "NetCat (nc) command not found. Skipping server detection."
fi

# Check if Python is installed
if command -v python3 &> /dev/null; then
  echo "Python version: $(python3 --version)"
  
  # Check if ChromaDB is installed
  python3 -c "import chromadb" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "ChromaDB is not installed. The application will run in fallback mode."
    echo "To install ChromaDB, run: pip install chromadb"
    echo "Running in fallback mode..."
  else
    echo "ChromaDB is installed. Running with full functionality."
  fi
else
  echo "Python 3 not found. The application may not work correctly."
  SKIP_PYTHON=true
fi

# Set environment variable to skip Python backend if requested
if [ "$SKIP_PYTHON" = true ]; then
  echo "Skipping Python backend startup in Node.js server..."
  export SKIP_PYTHON_BACKEND=true
fi

# Check if package.json exists
if [ ! -f "package.json" ]; then
  echo "ERROR: package.json not found in the current directory!"
  exit 1
fi

# Start the server
echo "Starting Node.js server using npm run dev..."
npm run dev