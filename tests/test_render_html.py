from triage import render_summary_entry_html, validate_rendered_html


def test_render_summary_entry_html_format():
    payload = {
        "title": "Readable title",
        "url": "https://example.com/post",
        "word_count": 880,
        "reading_time_minutes": 4,
        "summary": "A concise summary of the article.",
        "key_takeaways": ["One", "Two", "Three"],
        "suggested_next_step": "Save for later",
    }

    html = render_summary_entry_html(payload)

    validate_rendered_html(html)
    assert "<strong>Readable title</strong>" in html
    assert '<a href="https://example.com/post">https://example.com/post</a>' in html
    assert "Length: 880 words" in html
    assert "Reading time: 4 min" in html
    assert "<li>One</li>" in html
    assert "Suggested next step:" not in html
    assert html.strip().endswith("<hr/>")
