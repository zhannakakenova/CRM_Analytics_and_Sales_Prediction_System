#!/usr/bin/env sh
set -e

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
python -m app.dashboard.gradio_app
