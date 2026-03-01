#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import shutil
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
DEFAULT_MODEL_ID = "meta/llama-3.3-70b"
DEFAULT_NOTES_WARMUP_SECONDS = 8

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Safari/605.1.15 reading-triage/0.1"
)
# Keep prompts well below small local context windows (e.g., 4k).
MAX_ARTICLE_CHARS = 6_000
READING_WPM = 220
BLOCK_PAGE_MAX_WORDS = 260
BLOCK_PAGE_PHRASES = (
    "javascript is disabled",
    "enable javascript",
    "you need to enable javascript",
    "please enable javascript to proceed",
    "a required part of this site couldn't load",
    "a required part of this site couldn’t load",
    "checking if the site connection is secure",
    "verify you are human",
    "enable js",
)


class SummaryValidationError(Exception):
    """Raised when model output cannot be validated after retry."""


def resolve_inbox_note_name(note_name: str | None = None) -> str:
    return note_name or os.getenv("TRIAGE_INBOX_NOTE_NAME", DEFAULT_INBOX_NOTE_NAME)


def warm_up_notes_app(
    logger: logging.Logger | None = None,
    warmup_seconds: int | None = None,
) -> None:
    if warmup_seconds is None:
        raw = os.getenv("TRIAGE_NOTES_WARMUP_SECONDS", str(DEFAULT_NOTES_WARMUP_SECONDS)).strip()
        try:
            warmup_seconds = int(raw)
        except ValueError:
            warmup_seconds = DEFAULT_NOTES_WARMUP_SECONDS

    if warmup_seconds <= 0:
        return

    if logger:
        logger.info("Warming up Notes for %ds before inbox read", warmup_seconds)

    subprocess.run(
        ["open", "-gja", "Notes"],
        check=False,
        capture_output=True,
        text=True,
    )
    time.sleep(warmup_seconds)


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
    resolved_note_name = resolve_inbox_note_name(note_name)
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


def write_inbox_note_text(note_name: str | None, note_text: str) -> None:
    resolved_note_name = resolve_inbox_note_name(note_name)
    script = (
        'on run argv\n'
        'set noteName to item 1 of argv\n'
        'set bodyPath to item 2 of argv\n'
        'set newBody to (read POSIX file bodyPath as «class utf8»)\n'
        'tell application "Notes"\n'
        'set matches to (every note whose name is noteName)\n'
        'if (count of matches) is 0 then\n'
        'error "Note not found: " & noteName\n'
        "end if\n"
        "set body of (item 1 of matches) to newBody\n"
        "end tell\n"
        "end run"
    )
    temp_body_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as temp_file:
            temp_file.write(note_text)
            temp_body_path = Path(temp_file.name)

        argv = ["osascript", "-e", script, resolved_note_name, str(temp_body_path)]
        subprocess.run(argv, check=True, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to update Apple Notes inbox note '{resolved_note_name}'. "
            "Ensure Notes automation permissions are granted."
        ) from exc
    finally:
        if temp_body_path and temp_body_path.exists():
            temp_body_path.unlink(missing_ok=True)


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


def summarize_error_for_inbox(error_text: str, max_length: int = 140) -> str:
    text = " ".join(str(error_text).split())

    prefix = "Failed to produce valid summary JSON: "
    if text.startswith(prefix):
        text = text[len(prefix) :].strip()

    fallback_match = re.match(
        r"^Primary path failed \((.*)\); Playwright fallback failed \((.*)\)$",
        text,
    )
    if fallback_match:
        primary = fallback_match.group(1).strip()
        fallback = fallback_match.group(2).strip()
        text = f"Fetch failed: {primary}; fallback failed: {fallback}"

    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 3)].rstrip() + "..."


def _split_inbox_note_blocks(note_text: str) -> list[list[str]]:
    normalized = _normalize_note_text(note_text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line == "---":
            if current:
                blocks.append(current)
                current = []
            continue

        if line.startswith("Date:") and current and any(item.startswith("URL:") for item in current):
            blocks.append(current)
            current = []
        current.append(line)

    if current:
        blocks.append(current)
    return blocks


def _extract_url_from_block(block_lines: list[str]) -> str | None:
    for line in block_lines:
        match = re.match(r"^\s*URL:\s*(.+?)\s*$", line)
        if not match:
            continue
        extracted = _extract_http_url(match.group(1))
        if extracted:
            return extracted
    return None


def apply_triage_results_to_inbox_note(
    note_text: str,
    successful_urls: set[str],
    failed_errors: dict[str, str],
) -> str:
    blocks = _split_inbox_note_blocks(note_text)
    updated_blocks: list[list[str]] = []
    annotated_failures: set[str] = set()

    for block in blocks:
        block_url = _extract_url_from_block(block)
        if block_url and block_url in successful_urls:
            continue

        next_block = [line for line in block if not line.startswith("Error:")]
        if block_url and block_url in failed_errors:
            next_block.append(f"Error: {summarize_error_for_inbox(failed_errors[block_url])}")
            annotated_failures.add(block_url)
        updated_blocks.append(next_block)

    for url, error_text in failed_errors.items():
        if url in annotated_failures:
            continue
        updated_blocks.append(
            [
                f"URL: [{url}]",
                f"Error: {summarize_error_for_inbox(error_text)}",
            ]
        )

    if not updated_blocks:
        return ""

    lines: list[str] = []
    for block in updated_blocks:
        if not block:
            continue
        lines.append("---")
        lines.extend(block)
        lines.append("")
    return "\n".join(lines).strip()


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
    sentence_count = count_sentences(summary_text)
    if sentence_count < 2 or sentence_count > 4:
        raise SummaryValidationError("summary must contain 2-4 sentences")


def count_sentences(text: str) -> int:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return 0

    matches = re.findall(
        r".+?(?<!\d)[.!?](?!\d)[\"')\]]*(?=\s|$)",
        normalized,
    )
    if matches:
        return len(matches)
    return 1


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
        "title, url, word_count, reading_time_minutes, summary, key_takeaways. "
        "Constraints: summary must be 2-4 sentences, key_takeaways must have 3-7 concise items. "
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


def fetch_page_html_with_playwright(url: str, timeout: int = 30) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Playwright fallback unavailable. Install playwright and run 'playwright install chromium'."
        ) from exc

    timeout_ms = max(timeout * 1000, 5_000)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=USER_AGENT,
                ignore_https_errors=True,
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 15_000))
            except Exception:  # noqa: BLE001
                pass
            return page.content()
        finally:
            browser.close()


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


def fetch_and_extract_article_text(
    url: str,
    timeout: int = 30,
    logger: logging.Logger | None = None,
) -> str:
    primary_error: Exception | None = None

    try:
        html_content = fetch_page_html(url, timeout=timeout)
        extracted = extract_article_text(html_content, url)
        if is_probable_block_page_text(extracted):
            raise ValueError("Extraction returned a JavaScript/anti-bot placeholder page")
        return extracted
    except Exception as exc:  # noqa: BLE001
        primary_error = exc
        if logger:
            logger.info("Primary fetch/extract failed for %s; trying Playwright fallback", url)

    try:
        rendered_html = fetch_page_html_with_playwright(url, timeout=timeout)
        return extract_article_text(rendered_html, url)
    except Exception as fallback_exc:  # noqa: BLE001
        raise RuntimeError(
            f"Primary path failed ({primary_error}); Playwright fallback failed ({fallback_exc})"
        ) from fallback_exc


def is_probable_block_page_text(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return False
    words = len(normalized.split())
    if words > BLOCK_PAGE_MAX_WORDS:
        return False
    return any(phrase in normalized for phrase in BLOCK_PAGE_PHRASES)


def list_available_models(base_url: str) -> list[str]:
    response = requests.get(
        f"{base_url.rstrip('/')}/models",
        timeout=10,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("LM Studio /v1/models returned unexpected payload")
    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if model_id:
            models.append(str(model_id))
    return models


def _resolve_lms_cli_path() -> str:
    env_path = os.getenv("LMS_CLI_PATH")
    if env_path:
        resolved = str(Path(env_path).expanduser())
        if Path(resolved).exists():
            return resolved
        raise FileNotFoundError(f"LMS_CLI_PATH points to a missing file: {resolved}")

    detected = shutil.which("lms")
    if detected:
        return detected

    fallback = Path.home() / ".lmstudio" / "bin" / "lms"
    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError(
        "LM Studio CLI not found. Install LM Studio CLI or set LMS_CLI_PATH."
    )


def ensure_lmstudio_server_running(
    base_url: str,
    logger: logging.Logger | None = None,
    startup_wait_seconds: int = 20,
) -> None:
    try:
        list_available_models(base_url)
        return
    except (requests.ConnectionError, requests.Timeout) as exc:
        initial_error = exc

    if logger:
        logger.info("LM Studio server unreachable at %s; attempting to start via lms CLI", base_url)

    lms_path = _resolve_lms_cli_path()
    cmd = [lms_path, "server", "start"]
    port = urlparse(base_url).port
    if port:
        cmd.extend(["--port", str(port)])

    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        output = " | ".join(
            part.strip() for part in (completed.stdout, completed.stderr) if part and part.strip()
        )
        detail = output or f"exit code {completed.returncode}"
        raise RuntimeError(f"Failed to start LM Studio server: {detail}") from initial_error

    deadline = time.time() + max(1, startup_wait_seconds)
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            list_available_models(base_url)
            if logger:
                logger.info("LM Studio server is reachable at %s", base_url)
            return
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc
            time.sleep(1)

    raise RuntimeError(
        f"LM Studio server did not become reachable at {base_url} after startup attempt: {last_error}"
    ) from initial_error


def select_model_id(base_url: str, explicit_model: str | None = None) -> str:
    model_id = (explicit_model or "").strip()
    if not model_id:
        raise ValueError(
            "No model configured. Set --model or TRIAGE_LMSTUDIO_MODEL."
        )

    available_models = list_available_models(base_url)
    if model_id not in available_models:
        available = ", ".join(available_models) if available_models else "(none)"
        raise ValueError(
            f"Requested model '{model_id}' is not reachable via {base_url.rstrip('/')}/models. "
            f"Available models: {available}"
        )
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

    warm_up_notes_app(logger=logger)
    note_text = read_inbox_note_text(inbox_note_name)
    items = parse_inbox_note_items(note_text)
    now_ts = current_timestamp()
    for item in items:
        upsert_url(db_path, item["url"], item["title"], now_ts)

    pending = get_urls_to_summarize(db_path)
    if not pending:
        logger.info("No pending Reading Inbox items to summarize")
        return 0

    selected_model = (model or os.getenv("TRIAGE_LMSTUDIO_MODEL") or DEFAULT_MODEL_ID).strip()
    ensure_lmstudio_server_running(base_url=base_url, logger=logger)
    selected_model = select_model_id(
        base_url=base_url,
        explicit_model=selected_model,
    )
    logger.info("Using model: %s", selected_model)

    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv("TRIAGE_LMSTUDIO_API_KEY", "lm-studio"),
    )

    successful_entries: list[tuple[str, str]] = []
    failed_errors: dict[str, str] = {}

    for row in pending:
        url = row["url"]
        title = row["title"]

        try:
            article_text = fetch_and_extract_article_text(url, logger=logger)
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
            failed_errors[url] = error_text
            logger.error("Failed to summarize %s: %s", url, error_text)

        time.sleep(1)

    if not successful_entries and not failed_errors:
        logger.info("No successful summaries produced")
        return 0

    if successful_entries:
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

    if dry_run:
        logger.info("Dry run enabled: skipping Reading Inbox write")
    else:
        try:
            updated_inbox = apply_triage_results_to_inbox_note(
                note_text=note_text,
                successful_urls={url for url, _ in successful_entries},
                failed_errors=failed_errors,
            )
            write_inbox_note_text(inbox_note_name, updated_inbox)
            logger.info("Updated Apple Note: %s", resolve_inbox_note_name(inbox_note_name))
        except Exception as exc:  # noqa: BLE001
            logger.error("Reading Inbox update failed: %s", exc)
            return 1

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
        help=(
            "Model id override "
            f"(default: TRIAGE_LMSTUDIO_MODEL or '{DEFAULT_MODEL_ID}'; "
            "must be present in /v1/models)"
        ),
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
