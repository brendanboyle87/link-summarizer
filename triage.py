#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import trafilatura
from jsonschema import validate
from openai import OpenAI

APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "reading-triage"
DEFAULT_DB_PATH = APP_SUPPORT_DIR / "triage.db"
DEFAULT_INBOX_NOTE_NAME = "Reading Inbox"
DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"
DEFAULT_LOG_PATH = Path.home() / "Library" / "Logs" / "reading-triage.log"
DEFAULT_ERR_LOG_PATH = Path.home() / "Library" / "Logs" / "reading-triage.err"
DEFAULT_BASE_URL = "http://localhost:1234/v1"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Safari/605.1.15 reading-triage/0.1"
)
# Keep prompts well below small local context windows (e.g., 4k).
MAX_ARTICLE_CHARS = 6_000
READING_WPM = 220


class SummaryValidationError(Exception):
    """Raised when model output cannot be validated after retry."""


def configure_logging(
    log_path: Path = DEFAULT_LOG_PATH,
    err_path: Path = DEFAULT_ERR_LOG_PATH,
) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("reading-triage")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    log_handler = logging.FileHandler(log_path)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(formatter)

    err_handler = logging.FileHandler(err_path)
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)
    logger.addHandler(err_handler)
    return logger


def read_inbox_note_text(note_name: str | None = None) -> str:
    resolved_note_name = note_name or os.getenv("TRIAGE_INBOX_NOTE_NAME", DEFAULT_INBOX_NOTE_NAME)
    script = (
        'on run argv\n'
        'set noteName to item 1 of argv\n'
        'tell application "Notes"\n'
        'set matches to (every note whose name is noteName)\n'
        'if (count of matches) is 0 then\n'
        'error "Note not found: " & noteName\n'
        "end if\n"
        "return body of (item 1 of matches)\n"
        "end tell\n"
        "end run"
    )
    argv = ["osascript", "-e", script, resolved_note_name]
    try:
        completed = subprocess.run(argv, check=True, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to read Apple Notes inbox note '{resolved_note_name}'. "
            "Ensure the note exists and Notes automation permissions are granted."
        ) from exc
    return completed.stdout


def _normalize_note_text(note_text: str) -> str:
    normalized = html.unescape(note_text)
    normalized = re.sub(r"(?i)<br\s*/?>", "\n", normalized)
    normalized = re.sub(r"(?i)</(div|p|li|tr|h[1-6]|ul|ol)>", "\n", normalized)
    normalized = re.sub(r"(?i)<[^>]+>", "", normalized)
    return normalized


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _extract_http_url(url_field_value: str) -> str | None:
    candidate = url_field_value.strip()
    wrappers = {("[", "]"), ("<", ">"), ("(", ")"), ("{", "}"), ('"', '"'), ("'", "'")}
    changed = True
    while changed and candidate:
        changed = False
        for left, right in wrappers:
            if candidate.startswith(left) and candidate.endswith(right) and len(candidate) >= 2:
                candidate = candidate[1:-1].strip()
                changed = True

    if _is_http_url(candidate):
        return candidate

    match = re.search(r"https?://[^\s\]\[<>{}\"'()]+", candidate)
    if not match:
        return None
    extracted = match.group(0).rstrip(".,;:!?")
    return extracted if _is_http_url(extracted) else None


def parse_inbox_note_urls(note_text: str) -> list[str]:
    normalized = _normalize_note_text(note_text)
    found: list[str] = []
    seen: set[str] = set()
    for line in normalized.splitlines():
        match = re.match(r"^\s*URL:\s*(\S+)\s*$", line)
        if not match:
            continue
        extracted = _extract_http_url(match.group(1))
        if not extracted:
            continue
        if extracted in seen:
            continue
        seen.add(extracted)
        found.append(extracted)
    return found


def parse_inbox_note_items(note_text: str) -> list[dict[str, str]]:
    urls = parse_inbox_note_urls(note_text)
    return [{"url": url, "title": url} for url in urls]


def init_db(db_path: Path | str) -> None:
    resolved = Path(db_path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                summarized_at TEXT,
                status TEXT NOT NULL,
                error TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_summaries_status ON summaries(status)"
        )
        validate_db_schema(conn)
        conn.commit()


def validate_db_schema(conn: sqlite3.Connection) -> None:
    rows = conn.execute("PRAGMA table_info(summaries)").fetchall()
    expected = {
        "url": {"type": "TEXT", "pk": 1, "notnull": 0},
        "title": {"type": "TEXT", "pk": 0, "notnull": 1},
        "first_seen": {"type": "TEXT", "pk": 0, "notnull": 1},
        "summarized_at": {"type": "TEXT", "pk": 0, "notnull": 0},
        "status": {"type": "TEXT", "pk": 0, "notnull": 1},
        "error": {"type": "TEXT", "pk": 0, "notnull": 0},
    }
    if not rows:
        raise ValueError("summaries table is missing")

    actual = {
        row[1]: {"type": row[2].upper(), "notnull": row[3], "pk": row[5]}
        for row in rows
    }
    if set(actual) != set(expected):
        raise ValueError(f"Unexpected schema columns: {sorted(actual)}")

    for col, attrs in expected.items():
        actual_col = actual[col]
        if actual_col["type"] != attrs["type"]:
            raise ValueError(f"Column {col} expected type {attrs['type']}, got {actual_col['type']}")
        if actual_col["pk"] != attrs["pk"]:
            raise ValueError(f"Column {col} primary key mismatch")
        if actual_col["notnull"] != attrs["notnull"]:
            raise ValueError(f"Column {col} nullability mismatch")


def upsert_url(db_path: Path | str, url: str, title: str, first_seen: str) -> None:
    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        conn.execute(
            """
            INSERT INTO summaries(url, title, first_seen, summarized_at, status, error)
            VALUES(?, ?, ?, NULL, 'pending', NULL)
            ON CONFLICT(url) DO UPDATE SET
                title=excluded.title,
                status=CASE
                    WHEN summaries.summarized_at IS NULL THEN 'pending'
                    ELSE summaries.status
                END,
                error=CASE
                    WHEN summaries.summarized_at IS NULL THEN NULL
                    ELSE summaries.error
                END
            """,
            (url, title, first_seen),
        )
        conn.commit()


def get_urls_to_summarize(db_path: Path | str) -> list[dict[str, str]]:
    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT url, title, first_seen
            FROM summaries
            WHERE summarized_at IS NULL AND status = 'pending'
            ORDER BY first_seen ASC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def mark_summarized(db_path: Path | str, url: str, summarized_at: str) -> None:
    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        conn.execute(
            """
            UPDATE summaries
            SET summarized_at = ?, status = 'summarized', error = NULL
            WHERE url = ?
            """,
            (summarized_at, url),
        )
        conn.commit()


def mark_failed(db_path: Path | str, url: str, error: str) -> None:
    with sqlite3.connect(Path(db_path).expanduser()) as conn:
        conn.execute(
            """
            UPDATE summaries
            SET status = 'failed', error = ?
            WHERE url = ?
            """,
            (error, url),
        )
        conn.commit()


def load_schema(schema_path: Path | str = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    with Path(schema_path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_summary_payload(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    validate(instance=payload, schema=schema)

    takeaways = payload.get("key_takeaways", [])
    if not isinstance(takeaways, list) or not 3 <= len(takeaways) <= 7:
        raise SummaryValidationError("key_takeaways must contain between 3 and 7 items")

    summary_text = str(payload.get("summary", "")).strip()
    sentence_count = len([s for s in re.split(r"[.!?]+", summary_text) if s.strip()])
    if sentence_count < 2 or sentence_count > 4:
        raise SummaryValidationError("summary must contain 2-4 sentences")


def compute_word_metrics(text: str) -> tuple[int, int]:
    words = len(re.findall(r"\b\w+\b", text))
    reading_time = 0 if words == 0 else max(1, math.ceil(words / READING_WPM))
    return words, reading_time


def normalize_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)

    summary_text = str(normalized.get("summary", "")).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary_text) if s.strip()]
    if len(sentences) > 4:
        normalized["summary"] = " ".join(sentences[:4]).strip()
    elif len(sentences) < 2:
        base = summary_text.rstrip(".!?")
        if base:
            normalized["summary"] = (
                f"{base}. This article provides context to help prioritize your reading inbox."
            )
        else:
            normalized["summary"] = (
                "This article provides useful context. "
                "Read the source for complete details."
            )

    takeaways = normalized.get("key_takeaways")
    if not isinstance(takeaways, list):
        takeaways = []
    takeaways = [str(item).strip() for item in takeaways if str(item).strip()]
    if len(takeaways) > 7:
        takeaways = takeaways[:7]
    while len(takeaways) < 3:
        takeaways.append("Review the full article for additional detail.")
    normalized["key_takeaways"] = takeaways

    next_step = str(normalized.get("suggested_next_step", "")).strip()
    allowed = {"Read now", "Save for later", "Skip"}
    if next_step not in allowed:
        normalized["suggested_next_step"] = "Save for later"

    title = str(normalized.get("title", "")).strip()
    if not title:
        normalized["title"] = "Untitled article"

    return normalized


def build_messages(article_title: str, article_url: str, article_text: str, strict_retry: bool) -> list[dict[str, str]]:
    system_prompt = (
        "You summarize web articles for reading triage. "
        "Return JSON only, no markdown, no preamble."
    )
    user_prompt = (
        "Summarize the article and return strict JSON with keys: "
        "title, url, word_count, reading_time_minutes, summary, key_takeaways, suggested_next_step. "
        "Constraints: summary must be 2-4 sentences, key_takeaways must have 3-7 concise items, "
        "suggested_next_step must be exactly one of: Read now, Save for later, Skip. "
        f"Title: {article_title}\n"
        f"URL: {article_url}\n"
        "Article:\n"
        f"{article_text[:MAX_ARTICLE_CHARS]}"
    )
    if strict_retry:
        user_prompt += "\n\nReturn ONLY valid JSON matching the requested keys and constraints."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def summarize_article_with_retry(
    client: Any,
    model: str,
    article_title: str,
    article_url: str,
    article_text: str,
    schema: dict[str, Any],
    max_retries: int = 1,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        strict_retry = attempt > 0
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=build_messages(article_title, article_url, article_text, strict_retry),
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise SummaryValidationError("Model output must be a JSON object")

            payload["title"] = payload.get("title") or article_title
            payload["url"] = article_url

            words, reading_time = compute_word_metrics(article_text)
            payload["word_count"] = words
            payload["reading_time_minutes"] = reading_time
            payload = normalize_summary_payload(payload)

            validate_summary_payload(payload, schema)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise SummaryValidationError(f"Failed to produce valid summary JSON: {last_error}")


def fetch_page_html(url: str, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.text


def extract_article_text(html_content: str, url: str) -> str:
    text = trafilatura.extract(
        html_content,
        url=url,
        include_comments=False,
        include_tables=False,
    )
    if not text or not text.strip():
        raise ValueError("Extraction failed (possibly paywall, JS-heavy, or unsupported page)")
    return text.strip()


def select_model_id(base_url: str, explicit_model: str | None = None) -> str:
    if explicit_model:
        return explicit_model

    response = requests.get(
        f"{base_url.rstrip('/')}/models",
        timeout=10,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not data:
        raise ValueError("No models returned by LM Studio /v1/models")
    first = data[0]
    model_id = first.get("id") if isinstance(first, dict) else None
    if not model_id:
        raise ValueError("LM Studio model list did not include an id")
    return model_id


def render_summary_entry_html(summary_payload: dict[str, Any]) -> str:
    title = html.escape(str(summary_payload["title"]))
    url = html.escape(str(summary_payload["url"]))
    summary = html.escape(str(summary_payload["summary"]))
    word_count = int(summary_payload["word_count"])
    reading_time = int(summary_payload["reading_time_minutes"])

    takeaways = summary_payload.get("key_takeaways", [])
    li_html = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in takeaways)

    return (
        f"<strong>{title}</strong><br/>\n"
        f"<a href=\"{url}\">{url}</a><br/>\n"
        f"Length: {word_count} words - Reading time: {reading_time} min<br/>\n"
        f"<p>{summary}</p>\n"
        f"<ul>\n{li_html}\n</ul>\n"
        "<hr/>"
    )


def validate_rendered_html(rendered_html: str) -> None:
    required_markers = [
        "<strong>",
        "<a href=",
        "Length:",
        "Reading time:",
        "<ul>",
        "<li>",
        "<hr/>",
    ]
    missing = [marker for marker in required_markers if marker not in rendered_html]
    if missing:
        raise ValueError(f"Rendered HTML missing required markers: {missing}")


def upsert_daily_note(folder_name: str, note_title: str, html_to_append: str) -> None:
    script = r'''
on run argv
    set folderName to item 1 of argv
    set noteTitle to item 2 of argv
    set htmlPath to item 3 of argv
    set htmlChunk to (read POSIX file htmlPath as «class utf8»)

    tell application "Notes"
        if not (exists folder folderName) then
            make new folder with properties {name:folderName}
        end if

        set targetFolder to folder folderName
        set targetNote to missing value

        repeat with n in notes of targetFolder
            if name of n is noteTitle then
                set targetNote to n
                exit repeat
            end if
        end repeat

        if targetNote is missing value then
            make new note at targetFolder with properties {name:noteTitle, body:htmlChunk}
        else
            set body of targetNote to (body of targetNote) & htmlChunk
        end if
    end tell
end run
'''
    temp_html_path: Path | None = None
    temp_script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as html_tmp:
            html_tmp.write(html_to_append)
            temp_html_path = Path(html_tmp.name)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".applescript", delete=False) as script_tmp:
            script_tmp.write(script)
            temp_script_path = Path(script_tmp.name)

        argv = ["osascript", str(temp_script_path), folder_name, note_title, str(temp_html_path)]
        pid = os.posix_spawn("/usr/bin/osascript", argv, dict(os.environ))
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status):
            code = os.WEXITSTATUS(status)
            if code != 0:
                raise subprocess.CalledProcessError(code, argv)
        elif os.WIFSIGNALED(status):
            signal_number = os.WTERMSIG(status)
            raise subprocess.CalledProcessError(-signal_number, argv)
        else:
            raise RuntimeError("osascript did not exit cleanly")
    finally:
        if temp_html_path and temp_html_path.exists():
            temp_html_path.unlink(missing_ok=True)
        if temp_script_path and temp_script_path.exists():
            temp_script_path.unlink(missing_ok=True)


def current_timestamp() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def run_triage(
    inbox_note_name: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
    schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> int:
    logger = logger or configure_logging()

    init_db(db_path)
    schema = load_schema(schema_path)

    note_text = read_inbox_note_text(inbox_note_name)
    items = parse_inbox_note_items(note_text)
    now_ts = current_timestamp()
    for item in items:
        upsert_url(db_path, item["url"], item["title"], now_ts)

    pending = get_urls_to_summarize(db_path)
    if not pending:
        logger.info("No pending Reading Inbox items to summarize")
        return 0

    selected_model = select_model_id(
        base_url=base_url,
        explicit_model=model or os.getenv("TRIAGE_LMSTUDIO_MODEL"),
    )
    logger.info("Using model: %s", selected_model)

    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv("TRIAGE_LMSTUDIO_API_KEY", "lm-studio"),
    )

    successful_entries: list[tuple[str, str]] = []

    for row in pending:
        url = row["url"]
        title = row["title"]

        try:
            html_content = fetch_page_html(url)
            article_text = extract_article_text(html_content, url)
            summary = summarize_article_with_retry(
                client=client,
                model=selected_model,
                article_title=title,
                article_url=url,
                article_text=article_text,
                schema=schema,
                max_retries=1,
            )
            entry_html = render_summary_entry_html(summary)
            validate_rendered_html(entry_html)
            successful_entries.append((url, entry_html))
            logger.info("Prepared summary for %s", url)
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            mark_failed(db_path, url, error_text)
            logger.error("Failed to summarize %s: %s", url, error_text)

        time.sleep(1)

    if not successful_entries:
        logger.info("No successful summaries produced")
        return 0

    today = date.today().isoformat()
    note_title = f"Reading Triage — {today}"
    combined_html = "\n".join(chunk for _, chunk in successful_entries)

    if dry_run:
        logger.info("Dry run enabled: skipping Apple Notes write")
    else:
        try:
            upsert_daily_note("Reading Triage", note_title, combined_html)
            logger.info("Updated Apple Note: %s", note_title)
        except Exception as exc:  # noqa: BLE001
            error_text = f"Apple Notes write failed: {exc}"
            for url, _ in successful_entries:
                mark_failed(db_path, url, error_text)
            logger.error(error_text)
            return 1

    summarized_at = current_timestamp()
    for url, _ in successful_entries:
        mark_summarized(db_path, url, summarized_at)

    logger.info("Summarized %d URLs", len(successful_entries))
    return len(successful_entries)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Reading Inbox URLs into Apple Notes")
    parser.add_argument(
        "--inbox-note-name",
        default=None,
        help=(
            "Apple Notes inbox note name override "
            "(default: TRIAGE_INBOX_NOTE_NAME or 'Reading Inbox')"
        ),
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to sqlite DB",
    )
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to JSON schema",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="LM Studio OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id override (otherwise TRIAGE_LMSTUDIO_MODEL or /v1/models first item)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run everything except Apple Notes write",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = configure_logging()

    try:
        run_triage(
            inbox_note_name=args.inbox_note_name,
            db_path=args.db_path,
            schema_path=args.schema_path,
            base_url=args.base_url,
            model=args.model,
            dry_run=args.dry_run,
            logger=logger,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
