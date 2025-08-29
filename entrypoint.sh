#!/usr/bin/env bash
set -e


LOG_LEVEL=${LOG_LEVEL:-info}
export PYTHONUNBUFFERED=1
echo "Starting uvicorn on port ${PORT:-8000} with log-level ${LOG_LEVEL}"
uvicorn app:app \
  --host 0.0.0.0 \
  --port ${PORT:-8000} \
  --log-level ${LOG_LEVEL} \
  --access-log
