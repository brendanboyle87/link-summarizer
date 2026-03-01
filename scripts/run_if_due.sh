#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ONCE_SCRIPT="${PROJECT_ROOT}/scripts/run_once.sh"
APP_SUPPORT_DIR="$HOME/Library/Application Support/reading-triage"
STATE_FILE="${APP_SUPPORT_DIR}/launchd-last-run-date.txt"
LOCK_DIR="${APP_SUPPORT_DIR}/.launchd-run.lock"
TIMEOUT_FLAG="${APP_SUPPORT_DIR}/.launchd-run-timeout.flag"
LOG_PATH="$HOME/Library/Logs/reading-triage.log"
RUN_TIMEOUT_SECONDS=3600

mkdir -p "$APP_SUPPORT_DIR" "$(dirname "$LOG_PATH")"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_info() {
  printf '%s [INFO] %s\n' "$(timestamp)" "$1" >>"$LOG_PATH"
}

if [[ ! -x "$RUN_ONCE_SCRIPT" ]]; then
  log_info "launchd guard: missing run_once.sh at ${RUN_ONCE_SCRIPT}"
  exit 1
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log_info "launchd guard: another invocation is active, skipping"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

today="$(date +%F)"
hour_now="$(date +%H)"
if [[ "$hour_now" -lt 6 ]]; then
  log_info "launchd guard: before 06:00 local time, skipping"
  exit 0
fi

last_run_date=""
if [[ -f "$STATE_FILE" ]]; then
  last_run_date="$(tr -d ' \t\r\n' < "$STATE_FILE")"
fi

if [[ "$last_run_date" == "$today" ]]; then
  log_info "launchd guard: already ran for ${today}, skipping"
  exit 0
fi

log_info "launchd guard: due to run (last=${last_run_date:-none}), invoking run_once.sh"
"$RUN_ONCE_SCRIPT" &
run_pid=$!

caffeinate -dimsu -w "$run_pid" &
caffeinate_pid=$!

rm -f "$TIMEOUT_FLAG"
(
  sleep "$RUN_TIMEOUT_SECONDS"
  if kill -0 "$run_pid" 2>/dev/null; then
    touch "$TIMEOUT_FLAG"
    log_info "launchd guard: run_once.sh exceeded ${RUN_TIMEOUT_SECONDS}s timeout; terminating"
    kill -TERM "$run_pid" 2>/dev/null || true
    sleep 10
    if kill -0 "$run_pid" 2>/dev/null; then
      log_info "launchd guard: run_once.sh still active after TERM; killing"
      kill -KILL "$run_pid" 2>/dev/null || true
    fi
  fi
) &
watchdog_pid=$!

set +e
wait "$run_pid"
status=$?
set -e

kill "$watchdog_pid" 2>/dev/null || true
wait "$watchdog_pid" 2>/dev/null || true
wait "$caffeinate_pid" 2>/dev/null || true

if [[ "$status" -eq 0 ]]; then
  printf '%s\n' "$today" > "$STATE_FILE"
  log_info "launchd guard: run_once.sh succeeded for ${today}"
elif [[ -f "$TIMEOUT_FLAG" ]]; then
  log_info "launchd guard: run_once.sh timed out after ${RUN_TIMEOUT_SECONDS}s; will retry on next scheduled run/load"
  rm -f "$TIMEOUT_FLAG"
  exit 124
else
  log_info "launchd guard: run_once.sh failed with status ${status}; will retry on next scheduled run/load"
  exit "$status"
fi
