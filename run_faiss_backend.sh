#!/bin/bash

# Stop existing Python backend
pkill -f "python python_backend/run.py" || true

# Start the FAISS version of the Python backend
python python_backend/run_faiss.py
