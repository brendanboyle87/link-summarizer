import logging
from pathlib import Path

from triage import (
    apply_triage_results_to_inbox_note,
    run_triage,
    summarize_error_for_inbox,
)


def test_apply_triage_results_removes_success_and_marks_failed():
    note_text = """
---
Date: [Feb 24, 2026 at 06:10 AM]
URL: [https://example.com/success]

---
Date: [Feb 24, 2026 at 06:11 AM]
URL: [https://example.com/fail]
"""

    updated = apply_triage_results_to_inbox_note(
        note_text=note_text,
        successful_urls={"https://example.com/success"},
        failed_errors={
            "https://example.com/fail": "Failed to produce valid summary JSON: summary must contain 2-4 sentences"
        },
    )

    assert "https://example.com/success" not in updated
    assert "URL: [https://example.com/fail]" in updated
    assert "Error: summary must contain 2-4 sentences" in updated


def test_apply_triage_results_replaces_existing_error_line():
    note_text = """
---
Date: [Feb 24, 2026 at 06:11 AM]
URL: [https://example.com/fail]
Error: old message
"""

    updated = apply_triage_results_to_inbox_note(
        note_text=note_text,
        successful_urls=set(),
        failed_errors={"https://example.com/fail": "new message"},
    )

    assert "Error: old message" not in updated
    assert updated.count("Error: new message") == 1


def test_summarize_error_for_inbox_truncates_long_messages():
    long_error = "x" * 200
    condensed = summarize_error_for_inbox(long_error, max_length=40)
    assert condensed == ("x" * 37) + "..."


def test_run_triage_warms_up_notes_before_read(monkeypatch, tmp_path: Path):
    call_order: list[str] = []

    class _Logger:
        def info(self, *_args, **_kwargs):
            pass

        def error(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("triage.warm_up_notes_app", lambda logger=None: call_order.append("warm"))
    monkeypatch.setattr(
        "triage.read_inbox_note_text",
        lambda note_name=None: (call_order.append("read") or "URL: [https://example.com/a]"),
    )

    run_triage(
        db_path=tmp_path / "triage.db",
        schema_path=Path("schema.json"),
        logger=_Logger(),
    )

    assert call_order[:2] == ["warm", "read"]


def test_run_triage_updates_inbox_with_failure_annotations(monkeypatch, tmp_path: Path):
    updated = {}
    logger = logging.getLogger("test-run-triage-updates-inbox")
    logger.handlers = []
    logger.propagate = False
    logger.addHandler(logging.NullHandler())

    monkeypatch.setattr(
        "triage.read_inbox_note_text",
        lambda note_name=None: "URL: [https://example.com/fail]",
    )
    monkeypatch.setattr("triage.warm_up_notes_app", lambda logger=None: None)
    monkeypatch.setattr("triage.ensure_lmstudio_server_running", lambda **kwargs: None)
    monkeypatch.setattr("triage.select_model_id", lambda base_url, explicit_model=None: "meta/llama-3.3-70b")
    monkeypatch.setattr("triage.OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(
        "triage.fetch_and_extract_article_text",
        lambda url, logger=None: (_ for _ in ()).throw(ValueError("js disabled placeholder")),
    )
    monkeypatch.setattr(
        "triage.write_inbox_note_text",
        lambda note_name, note_text: updated.setdefault("text", note_text),
    )
    monkeypatch.setattr("triage.upsert_daily_note", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))

    result = run_triage(
        inbox_note_name="Reading Inbox",
        db_path=tmp_path / "triage.db",
        schema_path=Path("schema.json"),
        dry_run=False,
        logger=logger,
    )

    assert result == 0
    assert "Error: js disabled placeholder" in updated["text"]
