import pytest

from triage import fetch_and_extract_article_text


def test_fetch_and_extract_uses_primary_path_when_successful(monkeypatch):
    calls = {"primary_fetch": 0, "fallback_fetch": 0}

    def _primary_fetch(url: str, timeout: int = 30) -> str:
        calls["primary_fetch"] += 1
        return "<html>primary</html>"

    def _fallback_fetch(url: str, timeout: int = 30) -> str:
        calls["fallback_fetch"] += 1
        return "<html>fallback</html>"

    def _extract(html_content: str, url: str) -> str:
        return "primary-text" if "primary" in html_content else "fallback-text"

    monkeypatch.setattr("triage.fetch_page_html", _primary_fetch)
    monkeypatch.setattr("triage.fetch_page_html_with_playwright", _fallback_fetch)
    monkeypatch.setattr("triage.extract_article_text", _extract)

    result = fetch_and_extract_article_text("https://example.com")
    assert result == "primary-text"
    assert calls["primary_fetch"] == 1
    assert calls["fallback_fetch"] == 0


def test_fetch_and_extract_uses_playwright_when_primary_fetch_fails(monkeypatch):
    def _primary_fetch(url: str, timeout: int = 30) -> str:
        raise ValueError("403 forbidden")

    def _fallback_fetch(url: str, timeout: int = 30) -> str:
        return "<html>rendered</html>"

    def _extract(html_content: str, url: str) -> str:
        return "rendered-text"

    monkeypatch.setattr("triage.fetch_page_html", _primary_fetch)
    monkeypatch.setattr("triage.fetch_page_html_with_playwright", _fallback_fetch)
    monkeypatch.setattr("triage.extract_article_text", _extract)

    result = fetch_and_extract_article_text("https://example.com")
    assert result == "rendered-text"


def test_fetch_and_extract_uses_playwright_when_primary_extract_fails(monkeypatch):
    def _primary_fetch(url: str, timeout: int = 30) -> str:
        return "<html>primary</html>"

    def _fallback_fetch(url: str, timeout: int = 30) -> str:
        return "<html>rendered</html>"

    def _extract(html_content: str, url: str) -> str:
        if "rendered" in html_content:
            return "rendered-text"
        raise ValueError("Extraction failed")

    monkeypatch.setattr("triage.fetch_page_html", _primary_fetch)
    monkeypatch.setattr("triage.fetch_page_html_with_playwright", _fallback_fetch)
    monkeypatch.setattr("triage.extract_article_text", _extract)

    result = fetch_and_extract_article_text("https://example.com")
    assert result == "rendered-text"


def test_fetch_and_extract_raises_combined_error_when_both_paths_fail(monkeypatch):
    def _primary_fetch(url: str, timeout: int = 30) -> str:
        raise ValueError("primary failed")

    def _fallback_fetch(url: str, timeout: int = 30) -> str:
        raise RuntimeError("playwright unavailable")

    monkeypatch.setattr("triage.fetch_page_html", _primary_fetch)
    monkeypatch.setattr("triage.fetch_page_html_with_playwright", _fallback_fetch)

    with pytest.raises(RuntimeError) as exc_info:
        fetch_and_extract_article_text("https://example.com")

    msg = str(exc_info.value)
    assert "Primary path failed" in msg
    assert "playwright unavailable" in msg


def test_fetch_and_extract_uses_playwright_when_primary_is_js_placeholder(monkeypatch):
    def _primary_fetch(url: str, timeout: int = 30) -> str:
        return "<html>primary</html>"

    def _fallback_fetch(url: str, timeout: int = 30) -> str:
        return "<html>rendered</html>"

    def _extract(html_content: str, url: str) -> str:
        if "rendered" in html_content:
            return "Real rendered article body with details and data."
        return "JavaScript is disabled in your browser. Please enable JavaScript to proceed."

    monkeypatch.setattr("triage.fetch_page_html", _primary_fetch)
    monkeypatch.setattr("triage.fetch_page_html_with_playwright", _fallback_fetch)
    monkeypatch.setattr("triage.extract_article_text", _extract)

    result = fetch_and_extract_article_text("https://example.com")
    assert "Real rendered article body" in result
