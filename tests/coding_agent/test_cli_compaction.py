from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from agent.sessions import SessionManager
from coding_agent.cli import app
from coding_agent.config import ProviderConfig
from typer.testing import CliRunner
from upstream import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
from upstream.models import AssistantMessageEvent, Message
from upstream.providers.base import ModelProvider
from upstream.types import ToolSpec

runner = CliRunner()


class _CapturingProvider(ModelProvider):
    def __init__(self) -> None:
        self.seen_messages: list[list[Message]] = []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str | None = None,
        *,
        abort_event: asyncio.Event | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.seen_messages.append(list(messages))
        message = AssistantMessage(content=[TextContent(text="ok")], stopReason="stop")
        yield AssistantMessageEvent(type="start", partial=message)
        yield AssistantMessageEvent(type="done", reason="stop", message=message)


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


def test_cli_injects_compaction_summary_as_system_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ProviderConfig, "context_window", lambda self: 100)
    provider = _CapturingProvider()
    async def _build(self):
        return provider

    monkeypatch.setattr(ProviderConfig, "build", _build)
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
    final_turn_messages = provider.seen_messages[-1]
    assert final_turn_messages[0].role == "system"
    assert "Compacted conversation summary" in final_turn_messages[0].content
    assert final_turn_messages[1].role != "system"


def test_cli_compact_command_with_summary_flag_sets_from_hook(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    session_path = tmp_path / "session.jsonl"
    _seed_session(session_path)

    result = runner.invoke(
        app,
        ["compact", str(session_path), "--summary", "custom summary text"],
    )

    assert result.exit_code == 0
    records = _read_session(str(session_path))
    compactions = [r for r in records if r["type"] == "compaction"]
    assert len(compactions) == 1
    assert compactions[0]["summary"] == "custom summary text"
    assert compactions[0]["fromHook"] is True


def test_cli_compact_command_without_summary_flag_omits_from_hook(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    session_path = tmp_path / "session.jsonl"
    _seed_session(session_path)

    result = runner.invoke(app, ["compact", str(session_path)])

    assert result.exit_code == 0
    records = _read_session(str(session_path))
    compactions = [r for r in records if r["type"] == "compaction"]
    assert len(compactions) == 1
    assert "fromHook" not in compactions[0]


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
