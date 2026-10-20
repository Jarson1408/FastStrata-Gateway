#!/usr/bin/env bash
# Wrapper for the systemd user service.
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-$HOME/local-llm-gateway}"
PYTHON="${LOCAL_LLM_GATEWAY_PYTHON:-$INSTALL_DIR/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

cd "$INSTALL_DIR"
exec "$PYTHON" local_llm_gateway.py
