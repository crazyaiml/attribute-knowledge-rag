#!/usr/bin/env bash
# Stop the AK-RAG FastAPI server started by start.sh.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PID_FILE=".akrag.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "AK-RAG is not running (no $PID_FILE)."
  exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
  echo "AK-RAG is not running (stale PID $PID)."
  rm -f "$PID_FILE"
  exit 0
fi

kill "$PID"
for _ in $(seq 1 20); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 0.25
done

if kill -0 "$PID" 2>/dev/null; then
  echo "AK-RAG (PID $PID) did not stop gracefully, forcing..."
  kill -9 "$PID"
fi

rm -f "$PID_FILE"
echo "AK-RAG stopped."
