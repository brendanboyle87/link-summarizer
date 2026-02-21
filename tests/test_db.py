import sqlite3
from pathlib import Path

from triage import (
    get_urls_to_summarize,
    init_db,
    mark_failed,
    mark_summarized,
    upsert_url,
    validate_db_schema,
)


def test_init_insert_and_needs_summary_logic(tmp_path: Path):
    db_path = tmp_path / "triage.db"
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        validate_db_schema(conn)

    upsert_url(db_path, "https://example.com/a", "A", "2026-02-20T07:00:00")
    upsert_url(db_path, "https://example.com/a", "A newer", "2026-02-20T07:01:00")

    pending = get_urls_to_summarize(db_path)
    assert len(pending) == 1
    assert pending[0]["url"] == "https://example.com/a"
    assert pending[0]["title"] == "A newer"

    mark_summarized(db_path, "https://example.com/a", "2026-02-20T07:05:00")

    pending_after = get_urls_to_summarize(db_path)
    assert pending_after == []


def test_upsert_requeues_unsummarized_failed_url(tmp_path: Path):
    db_path = tmp_path / "triage.db"
    init_db(db_path)

    upsert_url(db_path, "https://example.com/retry", "Retry me", "2026-02-20T07:00:00")
    mark_failed(db_path, "https://example.com/retry", "temporary error")

    upsert_url(db_path, "https://example.com/retry", "Retry me v2", "2026-02-20T07:01:00")

    pending = get_urls_to_summarize(db_path)
    assert len(pending) == 1
    assert pending[0]["url"] == "https://example.com/retry"
    assert pending[0]["title"] == "Retry me v2"
