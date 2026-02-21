#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/run_once.sh"
MARKER_BEGIN="# reading-triage begin"
MARKER_END="# reading-triage end"
CRON_LINE="0 7 * * * ${RUN_SCRIPT}"
INBOX_NOTE_NAME="${TRIAGE_INBOX_NOTE_NAME:-Reading Inbox}"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  chmod +x "$RUN_SCRIPT"
fi

CURRENT_CRON="$(crontab -l 2>/dev/null || true)"
FILTERED="$(printf '%s\n' "$CURRENT_CRON" | awk -v begin="$MARKER_BEGIN" -v end="$MARKER_END" -v run_script="$RUN_SCRIPT" '
  $0 == begin { in_block=1; next }
  $0 == end { in_block=0; next }
  in_block == 1 { next }
  $0 == "# reading-triage" { next }
  index($0, run_script) > 0 { next }
  { print }
')"

if [[ -n "$FILTERED" ]]; then
  NEW_CRON="$(printf '%s\n%s\nTRIAGE_INBOX_NOTE_NAME=%s\n%s\n%s\n' \
    "$FILTERED" "$MARKER_BEGIN" "$INBOX_NOTE_NAME" "$CRON_LINE" "$MARKER_END")"
else
  NEW_CRON="$(printf '%s\nTRIAGE_INBOX_NOTE_NAME=%s\n%s\n%s\n' \
    "$MARKER_BEGIN" "$INBOX_NOTE_NAME" "$CRON_LINE" "$MARKER_END")"
fi

if ! printf '%s\n' "$NEW_CRON" | crontab -; then
  echo "Failed to install crontab. Ensure Terminal has permission to manage cron." >&2
  exit 1
fi

echo "Installed cron entry: $CRON_LINE"
echo "Using TRIAGE_INBOX_NOTE_NAME=$INBOX_NOTE_NAME"
