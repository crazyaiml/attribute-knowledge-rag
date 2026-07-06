#!/usr/bin/env bash
# Start the AK-RAG FastAPI server in the background.
# Config comes from .env (see .env.example) or environment overrides.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PID_FILE=".akrag.pid"
LOG_FILE="akrag.log"
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8080}"
LOG_LEVEL="${UVICORN_LOG_LEVEL:-warning}"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "AK-RAG is already running (PID $(cat "$PID_FILE"))."
  exit 0
fi

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "No .venv found — run: python3 -m venv .venv && pip install -e '.[claude,sentence]'" >&2
  exit 1
fi

PYTHONPATH=src nohup uvicorn akrag.main:app --host "$HOST" --port "$PORT" \
  --log-level "$LOG_LEVEL" --no-access-log \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "AK-RAG starting (PID $(cat "$PID_FILE")) on http://${HOST}:${PORT}"
echo "Logs: $LOG_FILE"
