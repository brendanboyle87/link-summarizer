from triage import parse_inbox_note_items


def test_parse_inbox_note_items_plain_text_multiple_urls():
    note = """
---
Date: 2026-02-21 09:14
URL: https://example.com/article

---
Date: 2026-02-21 10:02
URL: https://example.com/another
"""

    items = parse_inbox_note_items(note)

    assert items == [
        {"url": "https://example.com/article", "title": "https://example.com/article"},
        {"url": "https://example.com/another", "title": "https://example.com/another"},
    ]


def test_parse_inbox_note_items_optional_title_line():
    note = """
---
Date: 2026-02-21 09:14
Title: Some Article
URL: https://example.com/article
"""

    items = parse_inbox_note_items(note)

    assert items == [
        {"url": "https://example.com/article", "title": "https://example.com/article"},
    ]
