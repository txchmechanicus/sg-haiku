from __future__ import annotations

import json
from itertools import count
from pathlib import Path

from agent import AgentEvent
from agent.sessions import SessionManager
from upstream import (
    AssistantMessage,
    AssistantMessageEvent,
    SystemMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_json(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def load_jsonl(name: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_model(value) -> dict[str, object]:
    return value.model_dump(mode="json", exclude_none=True)


def sequential_ids():
    counter = count(1)

    def _next() -> str:
        return f"id-{next(counter)}"

    return _next


def with_fixed_timestamp(record: dict[str, object]) -> dict[str, object]:
    return {**record, "timestamp": "2026-01-01T00:00:00Z"}


def assistant_text_message() -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text="hello")], timestamp=123)


def assistant_tool_call_message() -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id="call-1", name="ls", arguments={"path": "."})],
        stopReason="toolUse",
        timestamp=123,
    )


def test_message_golden_fixtures() -> None:
    assert dump_model(UserMessage(content="hello", timestamp=123)) == load_json(
        "user_message.json"
    )
    assert dump_model(assistant_text_message()) == load_json("assistant_text_message.json")
    assert dump_model(assistant_tool_call_message()) == load_json(
        "assistant_tool_call_message.json"
    )
    assert dump_model(
        ToolResultMessage(
            toolCallId="call-1",
            toolName="ls",
            content=[TextContent(text="ok")],
            isError=False,
            timestamp=123,
        )
    ) == load_json("tool_result_message.json")
    assert dump_model(SystemMessage(content="hello", timestamp=123)) == load_json(
        "system_message.json"
    )


def test_assistant_message_event_golden_fixture() -> None:
    message = assistant_text_message()
    events = [
        AssistantMessageEvent(type="start", partial=message),
        AssistantMessageEvent(type="text_start", partial=message, contentIndex=0),
        AssistantMessageEvent(
            type="text_delta",
            partial=message,
            contentIndex=0,
            delta="hello",
        ),
        AssistantMessageEvent(
            type="text_end",
            partial=message,
            contentIndex=0,
            content="hello",
        ),
        AssistantMessageEvent(type="done", reason="stop", message=message),
    ]

    assert [dump_model(event) for event in events] == load_jsonl(
        "assistant_message_events.jsonl"
    )


def test_agent_event_golden_fixture() -> None:
    user = UserMessage(content="hello", timestamp=123)
    assistant = assistant_text_message()
    assistant_event = AssistantMessageEvent(
        type="text_delta",
        partial=assistant,
        contentIndex=0,
        delta="hello",
    )
    events = [
        AgentEvent.agent_start(),
        AgentEvent.message_start(user),
        AgentEvent.message_end(user),
        AgentEvent.turn_start(),
        AgentEvent.message_update(assistant, assistant_event),
        AgentEvent.message_end(assistant),
        AgentEvent.turn_end(assistant, []),
        AgentEvent.agent_end([user, assistant]),
    ]

    assert [dump_model(event) for event in events] == load_jsonl("agent_events.jsonl")


def test_session_header_and_message_entry_golden_fixtures(tmp_path: Path) -> None:
    manager = SessionManager.create(
        explicit_path=None,
        session_id="session-1",
        cwd=tmp_path,
        write_enabled=False,
        id_generator=sequential_ids(),
    )

    header = manager.header()
    header["timestamp"] = "2026-01-01T00:00:00Z"
    header["cwd"] = "/workspace"

    assert header == load_json("session_header.json")
    message_entry = manager.record_message(UserMessage(content="hello", timestamp=123))
    assert with_fixed_timestamp(message_entry) == load_json("session_message_entry.json")
    event_entry = manager.record_event(AgentEvent.agent_start())
    assert with_fixed_timestamp(event_entry) == load_json("session_event_entry.json")


def test_session_model_change_entry_golden_fixture(tmp_path: Path) -> None:
    manager = SessionManager.create(
        explicit_path=None,
        session_id="session-1",
        cwd=tmp_path,
        write_enabled=False,
        id_generator=sequential_ids(),
    )
    entry = manager.record_model_change(provider="mock", model_id="mock")
    assert with_fixed_timestamp(entry) == load_json("session_model_change_entry.json")


def test_session_thinking_level_change_entry_golden_fixture(tmp_path: Path) -> None:
    manager = SessionManager.create(
        explicit_path=None,
        session_id="session-1",
        cwd=tmp_path,
        write_enabled=False,
        id_generator=sequential_ids(),
    )
    entry = manager.record_thinking_level_change(thinking_level="high")
    assert with_fixed_timestamp(entry) == load_json("session_thinking_level_change_entry.json")


def test_session_compaction_entry_golden_fixture(tmp_path: Path) -> None:
    manager = SessionManager.create(
        explicit_path=None,
        session_id="session-1",
        cwd=tmp_path,
        write_enabled=False,
        id_generator=sequential_ids(),
    )
    entry = manager.record_compaction(
        summary="Compacted context.",
        first_kept_entry_id="entry-1",
        tokens_before=1000,
    )
    assert with_fixed_timestamp(entry) == load_json("session_compaction_entry.json")
