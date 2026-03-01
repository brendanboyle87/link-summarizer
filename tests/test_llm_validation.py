import json

import pytest

from triage import (
    MAX_ARTICLE_CHARS,
    SummaryValidationError,
    build_messages,
    summarize_article_with_retry,
    validate_summary_payload,
)


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _Response(self.outputs.pop(0))


class _Chat:
    def __init__(self, outputs):
        self.completions = _Completions(outputs)


class _Client:
    def __init__(self, outputs):
        self.chat = _Chat(outputs)


def _schema():
    return {
        "type": "object",
        "required": [
            "title",
            "url",
            "word_count",
            "reading_time_minutes",
            "summary",
            "key_takeaways",
        ],
        "properties": {
            "title": {"type": "string"},
            "url": {"type": "string"},
            "word_count": {"type": "integer"},
            "reading_time_minutes": {"type": "integer"},
            "summary": {"type": "string"},
            "key_takeaways": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "additionalProperties": False,
    }


def test_validate_summary_payload_rejects_invalid_payload():
    with pytest.raises(Exception):
        validate_summary_payload({"title": "missing required keys"}, _schema())


def test_summarize_retries_once_after_invalid_json_then_succeeds():
    client = _Client(
        [
            "not-json",
            json.dumps(
                    {
                        "title": "A",
                        "url": "https://example.com/a",
                        "word_count": 100,
                        "reading_time_minutes": 1,
                        "summary": "Sentence one. Sentence two.",
                        "key_takeaways": ["K1", "K2", "K3"],
                    }
                ),
            ]
    )

    result = summarize_article_with_retry(
        client=client,
        model="local-model",
        article_title="A",
        article_url="https://example.com/a",
        article_text="Body",
        schema=_schema(),
    )

    assert result["title"] == "A"
    assert client.chat.completions.calls == 2


def test_summarize_raises_after_second_invalid_output():
    client = _Client(["not-json", "still-not-json"])

    with pytest.raises(SummaryValidationError):
        summarize_article_with_retry(
            client=client,
            model="local-model",
            article_title="A",
            article_url="https://example.com/a",
            article_text="Body",
            schema=_schema(),
        )


def test_build_messages_truncates_article_text():
    long_text = "x" * (MAX_ARTICLE_CHARS + 50)

    messages = build_messages(
        article_title="A",
        article_url="https://example.com/a",
        article_text=long_text,
        strict_retry=False,
    )

    user_content = messages[1]["content"]
    assert long_text[:MAX_ARTICLE_CHARS] in user_content
    assert long_text[: MAX_ARTICLE_CHARS + 1] not in user_content


def test_summarize_normalizes_borderline_payload_values():
    client = _Client(
        [
            json.dumps(
                {
                    "title": "A",
                    "url": "https://example.com/a",
                    "word_count": 100,
                    "reading_time_minutes": 1,
                    "summary": "Single sentence only",
                    "key_takeaways": ["K1"],
                }
            )
        ]
    )

    result = summarize_article_with_retry(
        client=client,
        model="local-model",
        article_title="A",
        article_url="https://example.com/a",
        article_text="Body text for counting words",
        schema=_schema(),
    )

    sentence_count = len([s for s in result["summary"].split(".") if s.strip()])
    assert 2 <= sentence_count <= 4
    assert 3 <= len(result["key_takeaways"]) <= 7


def test_summarize_keeps_summary_within_four_sentences():
    client = _Client(
        [
            json.dumps(
                {
                    "title": "A",
                    "url": "https://example.com/a",
                    "word_count": 100,
                    "reading_time_minutes": 1,
                    "summary": "One. Two. Three. Four.",
                    "key_takeaways": ["K1", "K2", "K3"],
                }
            )
        ]
    )

    result = summarize_article_with_retry(
        client=client,
        model="local-model",
        article_title="A",
        article_url="https://example.com/a",
        article_text="Body text for counting words",
        schema=_schema(),
    )

    sentence_count = len([s for s in result["summary"].split(".") if s.strip()])
    assert sentence_count == 4


def test_validate_summary_payload_allows_decimals_without_overcounting_sentences():
    payload = {
        "title": "A",
        "url": "https://example.com/a",
        "word_count": 100,
        "reading_time_minutes": 1,
        "summary": (
            "Harness changes improved Terminal Bench 2.0 performance. "
            "The score rose from 52.8 to 66.5 in controlled tests. "
            "Teams used tracing to isolate regressions before changes. "
            "The update improved reliability while preserving quality."
        ),
        "key_takeaways": ["K1", "K2", "K3"],
    }

    validate_summary_payload(payload, _schema())
