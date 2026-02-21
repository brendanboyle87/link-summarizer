import pytest

from triage import parse_inbox_note_items, read_inbox_note_text


def test_parse_inbox_note_items_dedupes_urls_in_order():
    note = """
---
Date: 2026-02-21 09:14
URL: https://example.com/a

---
Date: 2026-02-21 09:15
URL: https://example.com/b

---
Date: 2026-02-21 09:16
URL: https://example.com/a
"""

    items = parse_inbox_note_items(note)

    assert items == [
        {"url": "https://example.com/a", "title": "https://example.com/a"},
        {"url": "https://example.com/b", "title": "https://example.com/b"},
    ]


def test_parse_inbox_note_items_ignores_malformed_entries():
    note = """
---
Date: 2026-02-21 09:14
URL:

---
Date: 2026-02-21 10:02
URL: not-a-url

---
Date: 2026-02-21 10:03
URL: ftp://example.com/unsupported

---
Date: 2026-02-21 10:04
URL: https://example.com/good
"""

    items = parse_inbox_note_items(note)

    assert items == [
        {"url": "https://example.com/good", "title": "https://example.com/good"},
    ]


def test_parse_inbox_note_items_handles_htmlish_note_body():
    note = """
<div>---</div>
<div>Date: 2026-02-21 09:14</div>
<div>URL: https://example.com/from-html-1</div>
<div><br></div>
<div>---</div>
<div>Date: 2026-02-21 10:02</div>
<div>URL: https://example.com/from-html-2</div>
"""

    items = parse_inbox_note_items(note)

    assert items == [
        {"url": "https://example.com/from-html-1", "title": "https://example.com/from-html-1"},
        {"url": "https://example.com/from-html-2", "title": "https://example.com/from-html-2"},
    ]


def test_parse_inbox_note_items_handles_bracket_wrapped_urls():
    note = """
<div>---</div>
<div>Date: [Feb 21, 2026 at 11:03 AM]</div>
<div>URL: [https://example.com/bracketed]</div>
<div>URL: [https://example.com/second?x=1]</div>
"""

    items = parse_inbox_note_items(note)

    assert items == [
        {"url": "https://example.com/bracketed", "title": "https://example.com/bracketed"},
        {"url": "https://example.com/second?x=1", "title": "https://example.com/second?x=1"},
    ]


def test_read_inbox_note_text_uses_env_override(monkeypatch):
    observed = {}

    def fake_run(cmd, check, capture_output, text):
        observed["cmd"] = cmd

        class Result:
            stdout = "note body"
            stderr = ""

        return Result()

    monkeypatch.setenv("TRIAGE_INBOX_NOTE_NAME", "My Inbox")
    monkeypatch.setattr("triage.subprocess.run", fake_run)

    body = read_inbox_note_text()

    assert body == "note body"
    assert observed["cmd"][0] == "osascript"
    assert observed["cmd"][1] == "-e"
    assert observed["cmd"][3] == "My Inbox"


def test_read_inbox_note_text_missing_note_raises_clear_error(monkeypatch):
    def fake_run(cmd, check, capture_output, text):
        raise RuntimeError("script failed")

    monkeypatch.setattr("triage.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Reading Inbox"):
        read_inbox_note_text()
