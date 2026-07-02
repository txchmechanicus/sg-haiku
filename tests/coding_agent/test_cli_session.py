from __future__ import annotations

import json
from pathlib import Path

from coding_agent.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_writes_session_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--session", "session.jsonl"])
    records = [
        json.loads(line)
        for line in Path("session.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    messages = [record for record in records if record["type"] == "message"]
    events = [record for record in records if record["type"] == "event"]

    assert result.exit_code == 0
    assert records[0]["type"] == "session"
    assert records[0]["version"] == 3
    assert messages[0]["message"]["role"] == "user"
    assert messages[1]["message"]["role"] == "assistant"
    assert events[0]["event"]["type"] == "agent_start"
    assert events[-1]["event"]["type"] == "agent_end"


def test_cli_no_session_does_not_create_session_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--no-session"])

    assert result.exit_code == 0
    assert not Path(".haiku").exists()


def test_cli_session_id_is_used_in_header(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        ["hello", "--session-id", "fixed-id", "--session", "session.jsonl"],
    )
    header = json.loads(Path("session.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert result.exit_code == 0
    assert header["id"] == "fixed-id"


def test_cli_session_dir_controls_default_session_location(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--session-dir", "sessions"])

    assert result.exit_code == 0
    files = list(Path("sessions").glob("*.jsonl"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])["type"] == "session"


def test_cli_continue_appends_latest_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    first = runner.invoke(app, ["hello", "--session-dir", "sessions"])
    session_path = next(Path("sessions").glob("*.jsonl"))
    second = runner.invoke(app, ["again", "--session-dir", "sessions", "--continue"])

    records = [json.loads(line) for line in session_path.read_text(encoding="utf-8").splitlines()]
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert [record["type"] for record in records].count("session") == 1
    assert [record["message"]["role"] for record in records if record["type"] == "message"] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [record["event"]["type"] for record in records if record["type"] == "event"].count(
        "agent_start"
    ) == 2


def test_cli_resume_by_path_loads_context_and_appends(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["hello", "--session", "session.jsonl"])
    result = runner.invoke(app, ["again", "--resume", "session.jsonl"])

    records = [
        json.loads(line)
        for line in Path("session.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert result.exit_code == 0
    assert [record["type"] for record in records].count("session") == 1
    messages = [record for record in records if record["type"] == "message"]
    assert messages[-2]["message"]["content"] == "again"


def test_cli_resume_by_partial_id_can_write_new_session_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app,
        ["hello", "--session", "original.jsonl", "--session-id", "abcdef-session"],
    )
    result = runner.invoke(
        app,
        ["again", "--resume", "abcdef", "--session-dir", ".", "--session", "next.jsonl"],
    )

    header = json.loads(Path("next.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert result.exit_code == 0
    assert header["id"] != "abcdef-session"


def test_cli_fork_creates_session_with_parent_session(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app,
        ["hello", "--session", "original.jsonl", "--session-id", "parent-session"],
    )
    result = runner.invoke(
        app,
        [
            "again",
            "--fork",
            "original.jsonl",
            "--session",
            "fork.jsonl",
            "--session-id",
            "child-session",
        ],
    )

    header = json.loads(Path("fork.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert result.exit_code == 0
    assert header["id"] == "child-session"
    assert header["parentSession"] == "parent-session"


def test_cli_rejects_multiple_session_load_modes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["hello", "--continue", "--fork", "missing"])

    assert result.exit_code == 2
    assert "Use only one of --continue, --resume, or --fork" in result.stderr
