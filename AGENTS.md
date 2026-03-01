# Agent Instructions

## Input Source

- Default input is Apple Notes note `Reading Inbox`.
- Optional override: `TRIAGE_INBOX_NOTE_NAME`.
- In inbox note content:
  - Parse `URL:` lines.
  - `Title:` is optional and must not be required.
  - Deduplicate URLs while preserving order.
  - HTML-ish note bodies and bracket-wrapped URLs like `URL: [https://example.com]` must be supported.

## Run Once Workflow

- Preferred execution path: `./scripts/run_once.sh`.
- `run_once.sh` should only run `triage.py` with existing defaults/log paths.
- Logs:
  - `~/Library/Logs/reading-triage.log`
  - `~/Library/Logs/reading-triage.err`

## Cron Workflow

- Install managed cron entry with `./scripts/install_cron.sh`.
- Script installs a 7:00 AM daily run and sets `TRIAGE_INBOX_NOTE_NAME` in the managed block.
- If note lookup fails, run must exit non-zero with a clear error message.

## Engineering Constraints

- Keep LM Studio integration unchanged unless required for compatibility.
- Keep DB path and schema stable:
  - `~/Library/Application Support/reading-triage/triage.db`
  - `schema.json` should not change unless explicitly required.
- Keep Apple Notes daily digest output format stable except explicitly requested changes.

## Test Workflow

- Use TDD for behavior changes.
- Run tests with `uv run pytest`.
- Keep the suite green after each meaningful change.
