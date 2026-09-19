#!/usr/bin/env bash
# Create a Python venv with NEURON's build-time and runtime dependencies.
set -euo pipefail
source "$(dirname "$0")/config.sh"

mkdir -p "$TOP/logs"
[ -d "$VENV" ] || "$PYTHON_BASE" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install -r "$TOP/requirements.txt"
"$VENV/bin/pip" freeze > "$TOP/logs/pip-freeze.txt"
