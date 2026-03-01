#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_GUARD_SCRIPT="${PROJECT_ROOT}/scripts/run_if_due.sh"
RUN_ONCE_SCRIPT="${PROJECT_ROOT}/scripts/run_once.sh"
LABEL="com.readingtriage.daily"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG_PATH="$HOME/Library/Logs/reading-triage.log"
ERR_PATH="$HOME/Library/Logs/reading-triage.err"
INBOX_NOTE_NAME="${TRIAGE_INBOX_NOTE_NAME:-Reading Inbox}"

if [[ ! -f "$RUN_GUARD_SCRIPT" ]]; then
  echo "Missing guard script: $RUN_GUARD_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$RUN_ONCE_SCRIPT" ]]; then
  echo "Missing run script: $RUN_ONCE_SCRIPT" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$(dirname "$LOG_PATH")"
chmod +x "$RUN_GUARD_SCRIPT" "$RUN_ONCE_SCRIPT"

cat >"$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>${RUN_GUARD_SCRIPT}</string>
  </array>

  <key>EnvironmentVariables</key>
  <dict>
    <key>TRIAGE_INBOX_NOTE_NAME</key>
    <string>${INBOX_NOTE_NAME}</string>
  </dict>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>6</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>

  <key>RunAtLoad</key>
  <true/>

  <key>WorkingDirectory</key>
  <string>${PROJECT_ROOT}</string>

  <key>StandardOutPath</key>
  <string>${LOG_PATH}</string>
  <key>StandardErrorPath</key>
  <string>${ERR_PATH}</string>
</dict>
</plist>
EOF

plutil -lint "$PLIST_PATH" >/dev/null

TARGET_DOMAIN="gui/$(id -u)"
launchctl bootout "$TARGET_DOMAIN" "$PLIST_PATH" >/dev/null 2>&1 || true
launchctl bootstrap "$TARGET_DOMAIN" "$PLIST_PATH"
launchctl enable "${TARGET_DOMAIN}/${LABEL}"

echo "Installed launchd agent: ${LABEL}"
echo "Plist: ${PLIST_PATH}"
echo "Schedule: daily at 06:00 local (with launchd calendar coalescing) + RunAtLoad"
echo "Using TRIAGE_INBOX_NOTE_NAME=${INBOX_NOTE_NAME}"
