from __future__ import annotations

import json
from pathlib import Path

from coding_agent.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def _read_session(filename: str) -> list[dict]:
    return [json.loads(line) for line in Path(filename).read_text(encoding="utf-8").splitlines()]


def test_cli_writes_model_change_entry_at_run_start(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--session", "session.jsonl"])
    records = _read_session("session.jsonl")
    model_change = [r for r in records if r["type"] == "model_change"]
    types = [r["type"] for r in records]

    assert result.exit_code == 0
    assert len(model_change) == 1
    assert model_change[0]["provider"] == "mock"
    assert model_change[0]["modelId"] == "mock"
    assert types.index("model_change") < types.index("event")


def test_cli_writes_thinking_level_change_when_thinking_is_active(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--session", "session.jsonl", "--thinking", "high"])
    records = _read_session("session.jsonl")
    thinking = [r for r in records if r["type"] == "thinking_level_change"]

    assert result.exit_code == 0
    assert len(thinking) == 1
    assert thinking[0]["thinkingLevel"] == "high"


def test_cli_does_not_write_thinking_level_change_when_thinking_is_off(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--session", "session.jsonl", "--thinking", "off"])
    records = _read_session("session.jsonl")
    thinking = [r for r in records if r["type"] == "thinking_level_change"]

    assert result.exit_code == 0
    assert len(thinking) == 0


def test_cli_does_not_write_thinking_level_change_when_thinking_not_specified(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--session", "session.jsonl"])
    records = _read_session("session.jsonl")
    thinking = [r for r in records if r["type"] == "thinking_level_change"]

    assert result.exit_code == 0
    assert len(thinking) == 0
