#!/usr/bin/env bash
set -e

pip -q install -r requirements.txt
python -m app.uvicorn_runner --host 0.0.0.0 --port 8000
