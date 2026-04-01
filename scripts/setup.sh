#!/usr/bin/env bash
set -euo pipefail

echo "=== Raven RAG System Setup ==="

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "ERROR: Python 3.11+ is required"; exit 1; }
command -v ollama >/dev/null 2>&1 || echo "WARNING: Ollama not found. Install from https://ollama.com"

# Create .env from example if not exists
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[+] Created .env from .env.example — edit to customize"
fi

# Create data directories
mkdir -p data/{uploads,cache,chroma}

# Create virtual environment
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "[+] Created virtual environment"
fi

# Activate and install
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
echo "[+] Installed Python dependencies"

# Check Ollama and pull default model
if command -v ollama >/dev/null 2>&1; then
    echo "[*] Pulling default LLM model (mistral)..."
    ollama pull mistral
    echo "[+] Model pulled"
else
    echo "[!] Ollama not installed — install it and run: ollama pull mistral"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "No Docker required! ChromaDB runs embedded in Python."
echo ""
echo "Start the backend (terminal 1):"
echo "  source .venv/bin/activate"
echo "  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Start the frontend (terminal 2):"
echo "  source .venv/bin/activate"
echo "  streamlit run ui/app.py"
echo ""
echo "Endpoints:"
echo "  Frontend:    http://localhost:8501"
echo "  API docs:    http://localhost:8000/docs"
echo "  Health:      http://localhost:8000/health"
