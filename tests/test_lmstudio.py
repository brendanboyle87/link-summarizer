from types import SimpleNamespace

import pytest
import requests

from triage import (
    DEFAULT_MODEL_ID,
    ensure_lmstudio_server_running,
    select_model_id,
)


def test_select_model_id_returns_requested_model_when_available(monkeypatch):
    monkeypatch.setattr("triage.list_available_models", lambda base_url: ["a", "b", DEFAULT_MODEL_ID])
    assert select_model_id("http://localhost:1234/v1", DEFAULT_MODEL_ID) == DEFAULT_MODEL_ID


def test_select_model_id_raises_clear_error_when_model_missing(monkeypatch):
    monkeypatch.setattr("triage.list_available_models", lambda base_url: ["a", "b"])
    with pytest.raises(ValueError) as exc_info:
        select_model_id("http://localhost:1234/v1", "missing-model")
    msg = str(exc_info.value)
    assert "missing-model" in msg
    assert "Available models" in msg


def test_ensure_server_does_not_start_when_already_reachable(monkeypatch):
    calls = {"list": 0, "start": 0}

    def _list(base_url: str):
        calls["list"] += 1
        return [DEFAULT_MODEL_ID]

    def _start(*args, **kwargs):
        calls["start"] += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("triage.list_available_models", _list)
    monkeypatch.setattr("triage.subprocess.run", _start)
    monkeypatch.setattr("triage._resolve_lms_cli_path", lambda: "lms")

    ensure_lmstudio_server_running("http://localhost:1234/v1")
    assert calls["list"] == 1
    assert calls["start"] == 0


def test_ensure_server_starts_when_unreachable_then_recovers(monkeypatch):
    calls = {"list": 0}

    def _list(base_url: str):
        calls["list"] += 1
        if calls["list"] == 1:
            raise requests.ConnectionError("connection refused")
        return [DEFAULT_MODEL_ID]

    monkeypatch.setattr("triage.list_available_models", _list)
    monkeypatch.setattr(
        "triage.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="started", stderr=""),
    )
    monkeypatch.setattr("triage._resolve_lms_cli_path", lambda: "lms")
    monkeypatch.setattr("triage.time.sleep", lambda s: None)

    ensure_lmstudio_server_running("http://localhost:1234/v1", startup_wait_seconds=1)
    assert calls["list"] >= 2


def test_ensure_server_raises_when_start_command_fails(monkeypatch):
    monkeypatch.setattr(
        "triage.list_available_models",
        lambda base_url: (_ for _ in ()).throw(requests.ConnectionError("connection refused")),
    )
    monkeypatch.setattr(
        "triage.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="oops", stderr="bad"),
    )
    monkeypatch.setattr("triage._resolve_lms_cli_path", lambda: "lms")
    monkeypatch.setattr("triage.time.sleep", lambda s: None)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_lmstudio_server_running("http://localhost:1234/v1", startup_wait_seconds=1)

    assert "Failed to start LM Studio server" in str(exc_info.value)
