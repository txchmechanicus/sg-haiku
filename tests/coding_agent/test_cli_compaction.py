from __future__ import annotations

import json
from pathlib import Path

from agent.sessions import SessionManager
from coding_agent.cli import app
from coding_agent.config import ProviderConfig
from typer.testing import CliRunner
from upstream import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage

runner = CliRunner()


def _read_session(filename: str) -> list[dict]:
    return [json.loads(line) for line in Path(filename).read_text(encoding="utf-8").splitlines()]


def _seed_session(path: Path) -> None:
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=path.parent)
    for i in range(6):
        manager.record_message(UserMessage(content=f"question {i} " + "x" * 200, timestamp=i))
        manager.record_message(
            AssistantMessage(
                content=[TextContent(text=f"answer {i} " + "y" * 200)],
                stopReason="stop",
                timestamp=i,
            )
        )


def test_cli_compacts_session_when_over_threshold(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ProviderConfig, "context_window", lambda self: 100)
    session_path = tmp_path / "session.jsonl"
    _seed_session(session_path)

    result = runner.invoke(
        app,
        [
            "one more question",
            "--resume",
            str(session_path),
            "--compaction-reserve-tokens",
            "10",
            "--compaction-keep-tokens",
            "20",
        ],
    )

    assert result.exit_code == 0
    records = _read_session(str(session_path))
    message_ids = {r["id"] for r in records if r["type"] == "message"}
    compactions = [r for r in records if r["type"] == "compaction"]
    assert len(compactions) == 1
    assert compactions[0]["summary"]
    assert compactions[0]["firstKeptEntryId"] in message_ids


def test_cli_compaction_records_touched_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ProviderConfig, "context_window", lambda self: 100)
    session_path = tmp_path / "session.jsonl"
    manager = SessionManager.create(
        explicit_path=session_path, session_id="session-1", cwd=tmp_path
    )
    for i in range(6):
        manager.record_message(UserMessage(content=f"question {i} " + "x" * 200, timestamp=i))
        manager.record_message(
            AssistantMessage(
                content=[ToolCall(id=f"call-{i}", name="read", arguments={"path": f"file{i}.py"})],
                stopReason="toolUse",
                timestamp=i,
            )
        )
        manager.record_message(
            ToolResultMessage(
                toolCallId=f"call-{i}",
                toolName="read",
                content=[TextContent(text=f"contents {i} " + "y" * 200)],
                isError=False,
                timestamp=i,
            )
        )

    result = runner.invoke(
        app,
        [
            "one more question",
            "--resume",
            str(session_path),
            "--compaction-reserve-tokens",
            "10",
            "--compaction-keep-tokens",
            "20",
        ],
    )

    assert result.exit_code == 0
    records = _read_session(str(session_path))
    compactions = [r for r in records if r["type"] == "compaction"]
    assert len(compactions) == 1
    details = compactions[0].get("details")
    assert details is not None
    assert details["readFiles"]
    assert all(path.startswith("file") for path in details["readFiles"])


def test_cli_no_compaction_flag_disables_compaction(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ProviderConfig, "context_window", lambda self: 100)
    session_path = tmp_path / "session.jsonl"
    _seed_session(session_path)

    result = runner.invoke(
        app,
        [
            "one more question",
            "--resume",
            str(session_path),
            "--compaction-reserve-tokens",
            "10",
            "--compaction-keep-tokens",
            "20",
            "--no-compaction",
        ],
    )

    assert result.exit_code == 0
    records = _read_session(str(session_path))
    compactions = [r for r in records if r["type"] == "compaction"]
    assert len(compactions) == 0
