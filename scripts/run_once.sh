#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
LOG_PATH="$HOME/Library/Logs/reading-triage.log"
ERR_PATH="$HOME/Library/Logs/reading-triage.err"

mkdir -p "$(dirname "$LOG_PATH")"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing virtualenv python at $VENV_PYTHON" >&2
  echo "Create it with: python3 -m venv ${PROJECT_ROOT}/.venv && ${PROJECT_ROOT}/.venv/bin/pip install -r ${PROJECT_ROOT}/requirements.txt" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

"$VENV_PYTHON" "$PROJECT_ROOT/triage.py" "$@" >>"$LOG_PATH" 2>>"$ERR_PATH"
