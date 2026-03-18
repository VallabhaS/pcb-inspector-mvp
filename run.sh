#!/usr/bin/env bash
# Launch both the FastAPI backend and Streamlit frontend.
# Usage: bash run.sh

set -e

echo "=== PCB Inspector MVP ==="
echo ""

# Start FastAPI in the background
echo "[1/2] Starting FastAPI backend on http://127.0.0.1:8000 ..."
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload &
API_PID=$!

# Give the API a moment to boot
sleep 2

# Start Streamlit in the foreground
echo "[2/2] Starting Streamlit UI on http://localhost:8501 ..."
streamlit run ui/streamlit_app.py --server.port 8501

# Clean up API when Streamlit exits
kill $API_PID 2>/dev/null
