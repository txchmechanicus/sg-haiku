from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent import AgentEvent
from agent.sessions import (
    SessionManager,
    find_sessions,
    latest_session,
    load_session,
    resolve_session_reference,
)
from upstream import AssistantMessage, TextContent, UserMessage


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_load_session_reads_message_entries(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(
        explicit_path=path,
        session_id="session-1",
        cwd=tmp_path,
    )
    manager.record_message(UserMessage(content="hello", timestamp=123))
    manager.record_message(
        AssistantMessage(content=[TextContent(text="hi")], stopReason="stop", timestamp=124)
    )

    loaded = load_session(path)

    assert loaded.session_id == "session-1"
    assert [message.role for message in loaded.messages] == ["user", "assistant"]


def test_load_session_validates_event_entries_but_uses_messages_for_context(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(
        explicit_path=path,
        session_id="session-1",
        cwd=tmp_path,
    )
    manager.record_event(AgentEvent.agent_start())
    manager.record_message(UserMessage(content="hello", timestamp=123))

    loaded = load_session(path)
    records = read_jsonl(path)

    assert [record["type"] for record in records] == ["session", "event", "message"]
    assert records[1]["event"]["type"] == "agent_start"
    assert [message.role for message in loaded.messages] == ["user"]


def test_load_session_ignores_unknown_entries(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session",
                        "version": 4,
                        "id": "session-1",
                        "timestamp": "2026-01-01T00:00:00Z",
                        "cwd": str(tmp_path),
                    }
                ),
                json.dumps({"type": "custom", "value": True}),
                json.dumps(
                    {
                        "type": "message",
                        "id": "id-1",
                        "parentId": None,
                        "message": UserMessage(
                            content="hello",
                            timestamp=123,
                        ).model_dump(mode="json", exclude_none=True),
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_session(path)

    assert [message.role for message in loaded.messages] == ["user"]


def test_latest_session_uses_newest_session_file(tmp_path: Path) -> None:
    older = tmp_path / "older.jsonl"
    newer = tmp_path / "newer.jsonl"
    SessionManager.create(explicit_path=older, session_id="older", cwd=tmp_path)
    SessionManager.create(explicit_path=newer, session_id="newer", cwd=tmp_path)

    older.touch()
    newer.touch()

    assert latest_session(tmp_path).session_id == "newer"


def test_resolve_session_reference_by_exact_and_partial_id(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    SessionManager.create(
        explicit_path=path,
        session_id="abcdef-session",
        cwd=tmp_path,
    )

    assert resolve_session_reference("abcdef-session", tmp_path).path == path
    assert resolve_session_reference("abcdef", tmp_path).path == path


def test_resolve_session_reference_reports_ambiguous_partial_id(tmp_path: Path) -> None:
    SessionManager.create(explicit_path=tmp_path / "one.jsonl", session_id="abc-one", cwd=tmp_path)
    SessionManager.create(explicit_path=tmp_path / "two.jsonl", session_id="abc-two", cwd=tmp_path)

    with pytest.raises(ValueError, match="Ambiguous session id"):
        resolve_session_reference("abc", tmp_path)


def test_append_session_preserves_existing_header(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    first = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    first.record_message(UserMessage(content="one", timestamp=123))
    loaded = load_session(path)

    appended = SessionManager.create(
        explicit_path=loaded.path,
        session_id=loaded.session_id,
        cwd=tmp_path,
        append=True,
        header=loaded.header,
    )
    appended.record_message(UserMessage(content="two", timestamp=124))

    records = read_jsonl(path)
    assert [record["type"] for record in records] == ["session", "message", "message"]
    assert records[0] == loaded.header


def test_fork_session_records_parent_session(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    forked = tmp_path / "forked.jsonl"
    SessionManager.create(explicit_path=original, session_id="parent-session", cwd=tmp_path)

    manager = SessionManager.create(
        explicit_path=forked,
        session_id="child-session",
        cwd=tmp_path,
        parent_session="parent-session",
    )

    assert manager.header()["parentSession"] == "parent-session"
    assert read_jsonl(forked)[0]["parentSession"] == "parent-session"


def test_find_sessions_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    assert find_sessions(tmp_path / "missing") == []


def test_load_session_ignores_model_change_and_thinking_level_change_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    manager.record_model_change(provider="mock", model_id="mock")
    manager.record_thinking_level_change(thinking_level="high")
    manager.record_message(UserMessage(content="hello", timestamp=123))

    loaded = load_session(path)
    records = read_jsonl(path)

    assert [record["type"] for record in records] == [
        "session",
        "model_change",
        "thinking_level_change",
        "message",
    ]
    assert len(loaded.messages) == 1
    assert loaded.messages[0].role == "user"


def test_load_session_applies_latest_compaction_entry(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    manager.record_message(UserMessage(content="one", timestamp=1))
    manager.record_message(
        AssistantMessage(content=[TextContent(text="reply one")], stopReason="stop", timestamp=2)
    )
    third = manager.record_message(UserMessage(content="two", timestamp=3))
    manager.record_compaction(
        summary="Summary up to message 2.",
        first_kept_entry_id=str(third["id"]),
        tokens_before=1000,
    )
    manager.record_message(
        AssistantMessage(content=[TextContent(text="reply two")], stopReason="stop", timestamp=4)
    )

    loaded = load_session(path)

    assert loaded.compaction_summary == "Summary up to message 2."
    assert [message.role for message in loaded.messages] == ["user", "assistant"]
    assert loaded.messages[0].content == "two"


def test_load_session_exposes_compaction_details(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    first = manager.record_message(UserMessage(content="one", timestamp=1))
    manager.record_compaction(
        summary="Summary.",
        first_kept_entry_id=str(first["id"]),
        tokens_before=100,
        details={"readFiles": ["a.py"], "modifiedFiles": []},
    )

    loaded = load_session(path)

    assert loaded.compaction_details == {"readFiles": ["a.py"], "modifiedFiles": []}


def test_load_session_uses_only_the_last_compaction_entry(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    message_records = [
        manager.record_message(UserMessage(content=f"msg-{i}", timestamp=i)) for i in range(4)
    ]
    manager.record_compaction(
        summary="First summary.",
        first_kept_entry_id=str(message_records[1]["id"]),
        tokens_before=500,
    )
    manager.record_message(UserMessage(content="msg-4", timestamp=4))
    manager.record_compaction(
        summary="Second summary.",
        first_kept_entry_id=str(message_records[3]["id"]),
        tokens_before=800,
    )

    loaded = load_session(path)

    assert loaded.compaction_summary == "Second summary."
    assert [message.content for message in loaded.messages] == ["msg-3", "msg-4"]


def test_record_leaf_change_rewinds_context_and_keeps_abandoned_branch_in_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    first = manager.record_message(UserMessage(content="one", timestamp=1))
    manager.record_message(
        AssistantMessage(content=[TextContent(text="abandoned reply")], timestamp=2)
    )
    manager.record_leaf_change(str(first["id"]), summary="Abandoned a wrong turn.")
    manager.record_message(
        AssistantMessage(content=[TextContent(text="retried reply")], timestamp=3)
    )

    loaded = load_session(path)
    records = read_jsonl(path)

    assert [record["type"] for record in records] == [
        "session",
        "message",
        "message",
        "branch_summary",
        "leaf",
        "message",
    ]
    assert [message.content for message in loaded.messages if message.role == "user"] == ["one"]
    assert [
        part.text
        for message in loaded.messages
        if message.role == "assistant"
        for part in message.content
    ] == ["retried reply"]


def test_load_session_replays_leaf_id_across_multiple_rewinds(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    manager = SessionManager.create(explicit_path=path, session_id="session-1", cwd=tmp_path)
    root = manager.record_message(UserMessage(content="root", timestamp=1))
    manager.record_message(UserMessage(content="branch-a", timestamp=2))
    manager.record_leaf_change(str(root["id"]))
    manager.record_message(UserMessage(content="branch-b", timestamp=3))
    manager.record_leaf_change(str(root["id"]))
    third = manager.record_message(UserMessage(content="branch-c", timestamp=4))

    loaded = load_session(path)

    assert [message.content for message in loaded.messages] == ["root", "branch-c"]
    assert loaded.leaf_id == third["id"]
