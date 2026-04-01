#!/usr/bin/env bash
set -euo pipefail

echo "=== Raven RAG - Starting ==="

echo "[*] Checking Ollama..."
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[+] Ollama is running"
else
    echo "[!] Ollama not reachable at localhost:11434"
    echo "    Start it with: ollama serve"
fi

echo ""
echo "No Docker needed — ChromaDB runs embedded in Python."
echo ""
echo "=== Start in two terminals ==="
echo ""
echo "Terminal 1 (Backend):"
echo "  source .venv/bin/activate"
echo "  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Terminal 2 (Frontend):"
echo "  source .venv/bin/activate"
echo "  streamlit run ui/app.py"
echo ""
echo "Endpoints:"
echo "  Frontend:  http://localhost:8501"
echo "  Backend:   http://localhost:8000"
echo "  API Docs:  http://localhost:8000/docs"
