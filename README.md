# Reading Triage (MVP)

Daily macOS automation that reads URLs from an Apple Notes inbox note, summarizes new URLs with LM Studio (OpenAI-compatible API), and appends all new entries into one Apple Notes note titled `Reading Triage — YYYY-MM-DD` in a `Reading Triage` folder.

## Requirements

- macOS with Apple Notes
- Python 3.11+
- LM Studio running local server (OpenAI-compatible API)
- Automation permissions for Terminal (or launchd shell) to control Notes
- iPhone Shortcut (or equivalent) that appends URL blocks into a note named `Reading Inbox`

## Setup

```bash
cd /Users/brendanboyle/repos/link-summarizer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

The Playwright browser install enables JS-render fallback fetching when direct HTTP + extraction fails.

## Reading Inbox Setup (Apple Notes)

1. Create an Apple Notes note named exactly `Reading Inbox`.
2. Configure your iPhone Shortcut to append text blocks to that note in this format:

```text
---
Date: 2026-02-21 09:14
URL: https://example.com/article
```

The parser extracts `URL:` lines, ignores malformed lines, tolerates HTML-ish Notes body content, and deduplicates URLs in original order.

## Inbox Note Override

Default inbox note name is `Reading Inbox`.

Override it with:

```bash
export TRIAGE_INBOX_NOTE_NAME="My Custom Inbox"
```

Or run once with CLI override:

```bash
uv run python triage.py --inbox-note-name "My Custom Inbox"
```

## LM Studio

1. Ensure LM Studio CLI is installed (`lms` command).
2. `triage.py` auto-starts the LM Studio server if `http://localhost:1234/v1` is unreachable.
3. Default model is:

```bash
meta/llama-3.3-70b
```

4. Optionally set model explicitly:

```bash
export TRIAGE_LMSTUDIO_MODEL="your-model-id"
```

or pass at runtime:

```bash
uv run python triage.py --model your-model-id
```

5. Verify models endpoint:

```bash
curl -s http://localhost:1234/v1/models
```

The configured model must appear in `GET /v1/models`; there is no fallback to the first listed model.

## Run Once

```bash
./scripts/run_once.sh
```

Or directly:

```bash
uv run python triage.py
```

Useful flags:

```bash
uv run python triage.py --dry-run
uv run python triage.py --db-path ~/Library/Application\ Support/reading-triage/triage.db
uv run python triage.py --inbox-note-name "Reading Inbox"
```

## Launchd Schedule (06:00 local)

Install idempotently:

```bash
./scripts/install_launchd.sh
```

Remove:

```bash
./scripts/remove_launchd.sh
```

The LaunchAgent runs `scripts/run_if_due.sh`, which:

- runs the real workflow via `scripts/run_once.sh`
- runs daily at 06:00 local (`StartCalendarInterval`)
- runs once when the agent is loaded (`RunAtLoad`)
- keeps the Mac awake for the active run via `caffeinate`
- enforces a 1-hour max runtime (kills stuck run, retries next scheduled run/load)
- enforces once-per-day execution after 06:00 using a local state file

`run_once.sh` uses the project venv Python and writes logs to:

- `~/Library/Logs/reading-triage.log`
- `~/Library/Logs/reading-triage.err`

## Troubleshooting

- Missing inbox note:
  - Ensure a note named `Reading Inbox` (or your override) exists in Apple Notes.
  - If it does not, `triage.py` exits non-zero with a clear note-read error.
- Notes permissions:
  - On first run, macOS should prompt for Apple Events access. Allow Terminal (and your shell/launchd context) to control Notes.
  - If permission was denied previously, re-enable in System Settings > Privacy & Security > Automation.

## Logs

```bash
tail -f ~/Library/Logs/reading-triage.log
tail -f ~/Library/Logs/reading-triage.err
```

## Tests

```bash
uv run pytest
```

## Data Storage

SQLite DB path:

- `~/Library/Application Support/reading-triage/triage.db`

Tracked columns:

- `url` (PK)
- `title`
- `first_seen`
- `summarized_at`
- `status`
- `error`
