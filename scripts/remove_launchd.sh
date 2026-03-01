#!/usr/bin/env bash
set -euo pipefail

LABEL="com.readingtriage.daily"
PLIST_PATH="$HOME/Library/LaunchAgents/${LABEL}.plist"
TARGET_DOMAIN="gui/$(id -u)"

if [[ -f "$PLIST_PATH" ]]; then
  launchctl bootout "$TARGET_DOMAIN" "$PLIST_PATH" >/dev/null 2>&1 || true
  rm -f "$PLIST_PATH"
  echo "Removed launchd agent plist: ${PLIST_PATH}"
else
  echo "No plist found at ${PLIST_PATH}"
fi

echo "If needed, verify with: launchctl print ${TARGET_DOMAIN}/${LABEL}"
